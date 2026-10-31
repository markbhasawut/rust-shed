/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is dual-licensed under either the MIT license found in the
 * LICENSE-MIT file in the root directory of this source tree or the Apache
 * License, Version 2.0 found in the LICENSE-APACHE file in the root directory
 * of this source tree. You may select, at your option, one of the
 * above-listed licenses.
 */

//! An in-memory stats backend, for builds that have no fb303 to export stats to.
//!
//! [`install`] registers it as the process's stats manager factory, and [`snapshot`] reads
//! every stat it tracks: the totals since the stat was created and the last minute, kept in
//! [`SLOTS`] time slots of [`SLOT`] each. Histogram and quantile stats also count their samples
//! in log-linear buckets, from which [`Distribution::quantile`] estimates quantiles within
//! 12.5%. Aggregation types, bucket configs, percentiles and intervals are ignored.
//!
//! Stats are merged by name: `define_stats!` creates a stat once per thread that uses it, and
//! every instance writes to the same entry. Writes are atomic additions that take a lock only
//! when a stat enters a new slot, and nothing has to call [`StatsManager::aggregate`].

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::PoisonError;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicI64;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use fbinit::FacebookInit;
use stats_traits::stat_types::BoxHistogram;
use stats_traits::stat_types::BoxLocalCounter;
use stats_traits::stat_types::BoxLocalHistogram;
use stats_traits::stat_types::BoxLocalTimeseries;
use stats_traits::stat_types::BoxSingletonCounter;
use stats_traits::stat_types::Counter;
use stats_traits::stat_types::Histogram;
use stats_traits::stat_types::SingletonCounter;
use stats_traits::stat_types::Timeseries;
use stats_traits::stats_manager::AggregationType;
use stats_traits::stats_manager::BoxStatsManager;
use stats_traits::stats_manager::BucketConfig;
use stats_traits::stats_manager::StatsManager;
use stats_traits::stats_manager::StatsManagerFactory;

use crate::noop_stats::Noop;

/// The length of a time slot.
pub const SLOT: Duration = Duration::from_secs(5);
/// The number of time slots per stat. The recent window is the current slot and the
/// `SLOTS - 1` before it: the last 55 to 60 seconds.
pub const SLOTS: usize = 12;
/// The most stats tracked. Stats created beyond it are not tracked.
pub const MAX_STATS: usize = 20_000;
/// The most histogram and quantile stats tracked, out of [`MAX_STATS`].
pub const MAX_DISTRIBUTIONS: usize = 2_000;

/// Four buckets per power of two; values from 2^41 up share the last.
const BUCKETS: usize = 160;
/// The id of a slot that was never written.
const UNUSED: u64 = u64::MAX;

/// Time since the registry's epoch.
type Clock = Arc<dyn Fn() -> Duration + Send + Sync>;

/// The registry of the installed backend, or `None` if `install` did not register it.
static GLOBAL: OnceLock<Option<Arc<Registry>>> = OnceLock::new();

/// Registers the in-memory backend as this process's stats manager factory, unless one is
/// already registered (for example because a stat was used first). Returns whether the
/// in-memory backend is the registered factory. Idempotent: later calls return the first
/// call's result without registering again.
pub fn install() -> bool {
    GLOBAL
        .get_or_init(|| {
            let registry = Arc::new(Registry::new(real_clock(), MAX_STATS, MAX_DISTRIBUTIONS));
            let factory = MemoryStatsFactory {
                registry: Arc::clone(&registry),
            };
            crate::try_register_stats_manager_factory(factory).then_some(registry)
        })
        .is_some()
}

/// Whether `install` succeeded. False before any call.
pub fn is_installed() -> bool {
    matches!(GLOBAL.get(), Some(Some(_)))
}

/// Every stat the backend tracks, or `None` unless `install` succeeded.
pub fn snapshot() -> Option<Snapshot> {
    GLOBAL.get()?.as_ref().map(|registry| registry.snapshot())
}

/// How a stat was created.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StatKind {
    /// `counter`.
    Counter,
    /// `timeseries`.
    Timeseries,
    /// `histogram`.
    Histogram,
    /// `quantile_stat`.
    Quantile,
    /// `singleton_counter`.
    Singleton,
}

/// Kinds that can share a name: a name created with one kind can be created again with
/// another kind of the same class.
#[derive(Clone, Copy, PartialEq, Eq)]
enum KindClass {
    Scalar,
    Distribution,
    Singleton,
}

impl StatKind {
    fn class(self) -> KindClass {
        match self {
            StatKind::Counter | StatKind::Timeseries => KindClass::Scalar,
            StatKind::Histogram | StatKind::Quantile => KindClass::Distribution,
            StatKind::Singleton => KindClass::Singleton,
        }
    }
}

/// Every stat the backend tracks, at one point in time.
#[derive(Clone, Debug)]
pub struct Snapshot {
    /// Time since `install`.
    pub uptime: Duration,
    /// What `StatSnapshot::recent` covers: 55 to 60 s, less right after start.
    pub window: Duration,
    /// Stat creations that were refused, so not tracked: over a limit, or a name reused with
    /// another kind class. A stat is created once per thread that uses it, so it can count
    /// more than once.
    pub dropped: u64,
    /// Sorted by name; names are unique.
    pub stats: Vec<StatSnapshot>,
}

impl Snapshot {
    /// The stat named `name`.
    pub fn get(&self, name: &str) -> Option<&StatSnapshot> {
        self.stats
            .binary_search_by(|stat| stat.name.as_str().cmp(name))
            .ok()
            .map(|index| &self.stats[index])
    }
}

/// One stat.
#[derive(Clone, Debug)]
pub struct StatSnapshot {
    /// The full name, prefix included.
    pub name: String,
    /// The kind it was first created with.
    pub kind: StatKind,
    /// Since the stat was created. For a singleton, `sum` is its value and `count` is 1,
    /// or 0 and 0 if it was never set.
    pub total: Window,
    /// The last `Snapshot::window`. Singletons: equal to `total`.
    pub recent: Window,
}

/// What a stat recorded over some time.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Window {
    /// The sum of the values added (wrapping).
    pub sum: i64,
    /// The number of samples added.
    pub count: u64,
    /// Histogram and quantile stats only.
    pub distribution: Option<Distribution>,
}

impl Window {
    /// `sum / count`, or `None` when `count` is 0.
    pub fn average(&self) -> Option<f64> {
        (self.count != 0).then(|| self.sum as f64 / self.count as f64)
    }

    /// `distribution.quantile(q)`, or `None` without a distribution.
    pub fn quantile(&self, q: f64) -> Option<i64> {
        self.distribution.as_ref()?.quantile(q)
    }
}

/// Mergeable sample counts in log-linear buckets: one bucket per value up to 7, then four
/// per power of two, so a bucket's midpoint is within 12.5% of every value in it. Values up
/// to 0 count as 0, and values from 2^41 up share the last bucket.
#[derive(Clone, PartialEq)]
pub struct Distribution {
    counts: Vec<u64>,
}

impl Distribution {
    /// An empty distribution.
    pub fn new() -> Self {
        Distribution {
            counts: vec![0; BUCKETS],
        }
    }

    /// Counts `value` `n` times.
    pub fn record(&mut self, value: i64, n: u64) {
        let count = &mut self.counts[bucket_index(value)];
        *count = count.saturating_add(n);
    }

    /// Adds `other`'s counts to these.
    pub fn merge(&mut self, other: &Distribution) {
        for (count, other) in self.counts.iter_mut().zip(&other.counts) {
            *count = count.saturating_add(*other);
        }
    }

    /// The number of samples.
    pub fn count(&self) -> u64 {
        self.counts
            .iter()
            .fold(0, |total, count| total.saturating_add(*count))
    }

    /// The `q` quantile (0.0..=1.0) as a bucket midpoint; `None` when empty.
    pub fn quantile(&self, q: f64) -> Option<i64> {
        let count = self.count();
        if count == 0 {
            return None;
        }
        let q = if q.is_nan() { 0.0 } else { q.clamp(0.0, 1.0) };
        let rank = ((q * count as f64).ceil() as u64).clamp(1, count);
        let mut seen = 0u64;
        self.counts.iter().enumerate().find_map(|(index, count)| {
            seen = seen.saturating_add(*count);
            (seen >= rank).then(|| bucket_midpoint(index))
        })
    }

    fn add_counts(&mut self, counts: impl Iterator<Item = u64>) {
        for (count, add) in self.counts.iter_mut().zip(counts) {
            *count = count.saturating_add(add);
        }
    }
}

impl Default for Distribution {
    fn default() -> Self {
        Distribution::new()
    }
}

impl fmt::Debug for Distribution {
    /// The non-empty buckets, by their smallest value.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(
                self.counts
                    .iter()
                    .enumerate()
                    .filter(|(_, count)| **count != 0)
                    .map(|(index, count)| (bucket_bounds(index).0, count)),
            )
            .finish()
    }
}

/// The bucket of `value`: 0 for values up to 0, the value itself up to 3, then
/// `((e - 1) << 2) + ((value >> (e - 2)) & 3)` with `e` the index of the highest bit set.
fn bucket_index(value: i64) -> usize {
    if value <= 0 {
        return 0;
    }
    let value = value as u64;
    if value <= 3 {
        return value as usize;
    }
    let e = u64::from(63 - value.leading_zeros());
    let index = ((e - 1) << 2) + ((value >> (e - 2)) & 3);
    (index as usize).min(BUCKETS - 1)
}

/// The smallest and largest value in bucket `index`. Bucket 0 also holds every negative
/// value, and the last bucket every value above its largest.
fn bucket_bounds(index: usize) -> (i64, i64) {
    if index < 4 {
        return (index as i64, index as i64);
    }
    let shift = (index >> 2) - 1;
    let step = (index & 3) as i64;
    ((4 + step) << shift, ((5 + step) << shift) - 1)
}

fn bucket_midpoint(index: usize) -> i64 {
    let (low, high) = bucket_bounds(index);
    low + (high - low) / 2
}

/// The slot period `now` falls in.
fn slot_id(now: Duration) -> u64 {
    now.as_secs() / SLOT.as_secs()
}

fn real_clock() -> Clock {
    let epoch = Instant::now();
    Arc::new(move || epoch.elapsed())
}

/// What a stat recorded in one slot period.
struct Slot {
    /// The slot period, or `UNUSED`. Released after the slot is cleared for it.
    id: AtomicU64,
    sum: AtomicI64,
    count: AtomicU64,
    /// Distributions only. Wraps after 2^32 samples of one bucket in one period.
    buckets: Option<Box<[AtomicU32]>>,
}

impl Slot {
    fn new(distribution: bool) -> Self {
        Slot {
            id: AtomicU64::new(UNUSED),
            sum: AtomicI64::new(0),
            count: AtomicU64::new(0),
            buckets: distribution.then(|| (0..BUCKETS).map(|_| AtomicU32::new(0)).collect()),
        }
    }

    fn clear(&self) {
        self.sum.store(0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
        for bucket in self.buckets.iter().flat_map(|buckets| buckets.iter()) {
            bucket.store(0, Ordering::Relaxed);
        }
    }
}

/// The data of a counter, timeseries, histogram or quantile stat.
struct Series {
    /// Indexed by slot period modulo `SLOTS`.
    slots: [Slot; SLOTS],
    sum: AtomicI64,
    count: AtomicU64,
    /// Distributions only.
    buckets: Option<Box<[AtomicU64]>>,
    /// Held while a slot is cleared for a new period, and while the slots are read.
    rotation: Mutex<()>,
}

impl Series {
    fn new(distribution: bool) -> Self {
        Series {
            slots: std::array::from_fn(|_| Slot::new(distribution)),
            sum: AtomicI64::new(0),
            count: AtomicU64::new(0),
            buckets: distribution.then(|| (0..BUCKETS).map(|_| AtomicU64::new(0)).collect()),
            rotation: Mutex::new(()),
        }
    }

    /// Adds `sum` and `count` at `now`, and, for a distribution, `count` samples of `sample`.
    fn record(&self, now: Duration, sum: i64, count: u32, sample: i64) {
        let slot = self.slot(slot_id(now));
        slot.sum.fetch_add(sum, Ordering::Relaxed);
        slot.count.fetch_add(u64::from(count), Ordering::Relaxed);
        self.sum.fetch_add(sum, Ordering::Relaxed);
        self.count.fetch_add(u64::from(count), Ordering::Relaxed);
        if let (Some(slot_buckets), Some(buckets)) = (&slot.buckets, &self.buckets) {
            let index = bucket_index(sample);
            slot_buckets[index].fetch_add(count, Ordering::Relaxed);
            buckets[index].fetch_add(u64::from(count), Ordering::Relaxed);
        }
    }

    /// The slot of period `id`, cleared first if it holds an older period. A slot that
    /// already holds a newer period (the writer read the clock long before) is kept.
    fn slot(&self, id: u64) -> &Slot {
        let slot = &self.slots[(id % SLOTS as u64) as usize];
        if slot.id.load(Ordering::Acquire) != id {
            let _rotation = self.rotation.lock().unwrap_or_else(PoisonError::into_inner);
            let current = slot.id.load(Ordering::Relaxed);
            if current == UNUSED || current < id {
                slot.clear();
                slot.id.store(id, Ordering::Release);
            }
        }
        slot
    }

    /// The totals, and the sums of the slots of periods `cur - SLOTS + 1` to `cur`.
    fn read(&self, cur: u64) -> (Window, Window) {
        let total = Window {
            sum: self.sum.load(Ordering::Relaxed),
            count: self.count.load(Ordering::Relaxed),
            distribution: self.buckets.as_ref().map(|buckets| {
                let mut distribution = Distribution::new();
                distribution.add_counts(buckets.iter().map(|count| count.load(Ordering::Relaxed)));
                distribution
            }),
        };
        let mut recent = Window {
            distribution: self.buckets.as_ref().map(|_| Distribution::new()),
            ..Window::default()
        };
        let oldest = cur.saturating_sub(SLOTS as u64 - 1);
        let _rotation = self.rotation.lock().unwrap_or_else(PoisonError::into_inner);
        for slot in &self.slots {
            let id = slot.id.load(Ordering::Acquire);
            if id < oldest || id > cur {
                continue;
            }
            recent.sum = recent.sum.wrapping_add(slot.sum.load(Ordering::Relaxed));
            recent.count = recent
                .count
                .wrapping_add(slot.count.load(Ordering::Relaxed));
            if let (Some(distribution), Some(buckets)) = (&mut recent.distribution, &slot.buckets) {
                distribution.add_counts(
                    buckets
                        .iter()
                        .map(|count| u64::from(count.load(Ordering::Relaxed))),
                );
            }
        }
        (total, recent)
    }
}

/// The data of a singleton counter.
#[derive(Default)]
struct Singleton {
    value: AtomicI64,
    set: AtomicBool,
}

impl Singleton {
    fn window(&self) -> Window {
        if !self.set.load(Ordering::Acquire) {
            return Window::default();
        }
        Window {
            sum: self.value.load(Ordering::Relaxed),
            count: 1,
            distribution: None,
        }
    }
}

enum Data {
    Series(Arc<Series>),
    Singleton(Arc<Singleton>),
}

struct Entry {
    kind: StatKind,
    data: Data,
}

#[derive(Default)]
struct Entries {
    by_name: BTreeMap<String, Arc<Entry>>,
    distributions: usize,
}

/// The stats of one backend, merged by name.
struct Registry {
    clock: Clock,
    max_stats: usize,
    max_distributions: usize,
    entries: Mutex<Entries>,
    /// Refused creations. A plain atomic: the backend must not use stats itself, as the stats
    /// crate calls it with the factory lock held.
    dropped: AtomicU64,
}

impl Registry {
    fn new(clock: Clock, max_stats: usize, max_distributions: usize) -> Self {
        Registry {
            clock,
            max_stats,
            max_distributions,
            entries: Mutex::new(Entries::default()),
            dropped: AtomicU64::new(0),
        }
    }

    /// The entry named `name`, created if new. `None`, counted as dropped, if the name
    /// exists with another kind class or a limit is reached.
    fn entry(&self, name: &str, kind: StatKind) -> Option<Arc<Entry>> {
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        match entries.by_name.get(name) {
            Some(entry) if entry.kind.class() == kind.class() => return Some(Arc::clone(entry)),
            Some(_) => {}
            None => {
                let distribution = kind.class() == KindClass::Distribution;
                if entries.by_name.len() < self.max_stats
                    && (!distribution || entries.distributions < self.max_distributions)
                {
                    let data = match kind.class() {
                        KindClass::Singleton => Data::Singleton(Arc::new(Singleton::default())),
                        _ => Data::Series(Arc::new(Series::new(distribution))),
                    };
                    let entry = Arc::new(Entry { kind, data });
                    entries.by_name.insert(name.to_owned(), Arc::clone(&entry));
                    entries.distributions += usize::from(distribution);
                    return Some(entry);
                }
            }
        }
        self.dropped.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// A handle that writes to the series `name`, or `None` if it is not tracked.
    fn series(&self, name: &str, kind: StatKind) -> Option<SeriesHandle> {
        match &self.entry(name, kind)?.data {
            Data::Series(series) => Some(SeriesHandle {
                series: Arc::clone(series),
                clock: Arc::clone(&self.clock),
            }),
            Data::Singleton(_) => None,
        }
    }

    /// A handle to the singleton `name`, or `None` if it is not tracked.
    fn singleton(&self, name: &str) -> Option<SingletonHandle> {
        match &self.entry(name, StatKind::Singleton)?.data {
            Data::Singleton(singleton) => Some(SingletonHandle(Arc::clone(singleton))),
            Data::Series(_) => None,
        }
    }

    fn snapshot(&self) -> Snapshot {
        let entries: Vec<(String, Arc<Entry>)> = self
            .entries
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .by_name
            .iter()
            .map(|(name, entry)| (name.clone(), Arc::clone(entry)))
            .collect();
        let uptime = (self.clock)();
        let cur = slot_id(uptime);
        let slot_start = Duration::from_secs(cur * SLOT.as_secs());
        let window = uptime.min(SLOT * (SLOTS as u32 - 1) + (uptime - slot_start));
        let stats = entries
            .into_iter()
            .map(|(name, entry)| {
                let (total, recent) = match &entry.data {
                    Data::Series(series) => series.read(cur),
                    Data::Singleton(singleton) => {
                        let total = singleton.window();
                        (total.clone(), total)
                    }
                };
                StatSnapshot {
                    name,
                    kind: entry.kind,
                    total,
                    recent,
                }
            })
            .collect();
        Snapshot {
            uptime,
            window,
            dropped: self.dropped.load(Ordering::Relaxed),
            stats,
        }
    }
}

/// Writes to a counter, timeseries, histogram or quantile stat.
struct SeriesHandle {
    series: Arc<Series>,
    clock: Clock,
}

impl SeriesHandle {
    fn record(&self, sum: i64, count: u32, sample: i64) {
        self.series.record((self.clock)(), sum, count, sample);
    }
}

impl Counter for SeriesHandle {
    fn increment_value(&self, value: i64) {
        self.record(value, 1, value);
    }
}

impl Timeseries for SeriesHandle {
    fn add_value(&self, value: i64) {
        self.record(value, 1, value);
    }

    fn add_value_aggregated(&self, value: i64, nsamples: u32) {
        self.record(value, nsamples, value);
    }
}

impl Histogram for SeriesHandle {
    fn add_value(&self, value: i64) {
        self.record(value, 1, value);
    }

    fn add_repeated_value(&self, value: i64, nsamples: u32) {
        self.record(value.wrapping_mul(i64::from(nsamples)), nsamples, value);
    }
}

struct SingletonHandle(Arc<Singleton>);

impl SingletonCounter for SingletonHandle {
    fn set_value(&self, _fb: FacebookInit, value: i64) {
        self.0.value.store(value, Ordering::Relaxed);
        self.0.set.store(true, Ordering::Release);
    }

    fn increment_value(&self, _fb: FacebookInit, value: i64) {
        self.0.value.fetch_add(value, Ordering::Relaxed);
        self.0.set.store(true, Ordering::Release);
    }

    fn get_value(&self, _fb: FacebookInit) -> Option<i64> {
        self.0
            .set
            .load(Ordering::Acquire)
            .then(|| self.0.value.load(Ordering::Relaxed))
    }
}

struct MemoryStatsManager {
    registry: Arc<Registry>,
}

impl StatsManager for MemoryStatsManager {
    fn aggregate(&self) {}

    fn create_counter(&self, name: &str) -> BoxLocalCounter {
        match self.registry.series(name, StatKind::Counter) {
            Some(handle) => Box::new(handle),
            None => Box::new(Noop),
        }
    }

    fn create_timeseries(
        &self,
        name: &str,
        _aggregation_types: &[AggregationType],
        _intervals: &[Duration],
    ) -> BoxLocalTimeseries {
        match self.registry.series(name, StatKind::Timeseries) {
            Some(handle) => Box::new(handle),
            None => Box::new(Noop),
        }
    }

    fn create_histogram(
        &self,
        name: &str,
        _aggregation_types: &[AggregationType],
        _conf: BucketConfig,
        _percentiles: &[u8],
    ) -> BoxLocalHistogram {
        match self.registry.series(name, StatKind::Histogram) {
            Some(handle) => Box::new(handle),
            None => Box::new(Noop),
        }
    }

    fn create_quantile_stat(
        &self,
        name: &str,
        _aggregation_types: &[AggregationType],
        _percentiles: &[f32],
        _intervals: &[Duration],
    ) -> BoxHistogram {
        match self.registry.series(name, StatKind::Quantile) {
            Some(handle) => Box::new(handle),
            None => Box::new(Noop),
        }
    }
}

struct MemoryStatsFactory {
    registry: Arc<Registry>,
}

impl StatsManagerFactory for MemoryStatsFactory {
    fn create(&self) -> BoxStatsManager {
        Box::new(MemoryStatsManager {
            registry: Arc::clone(&self.registry),
        })
    }

    fn create_singleton_counter(&self, name: &str) -> Option<BoxSingletonCounter> {
        let counter: BoxSingletonCounter = match self.registry.singleton(name) {
            Some(handle) => Box::new(handle),
            None => Box::new(Noop),
        };
        Some(counter)
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    /// A registry whose clock reads `now`, in milliseconds.
    fn test_registry(
        max_stats: usize,
        max_distributions: usize,
    ) -> (Arc<Registry>, Arc<AtomicU64>) {
        let now = Arc::new(AtomicU64::new(0));
        let clock: Clock = {
            let now = Arc::clone(&now);
            Arc::new(move || Duration::from_millis(now.load(Ordering::SeqCst)))
        };
        let registry = Arc::new(Registry::new(clock, max_stats, max_distributions));
        (registry, now)
    }

    fn manager_of(registry: &Arc<Registry>) -> MemoryStatsManager {
        MemoryStatsManager {
            registry: Arc::clone(registry),
        }
    }

    fn set_secs(now: &AtomicU64, secs: u64) {
        now.store(secs * 1000, Ordering::SeqCst);
    }

    fn read_stat(registry: &Registry, name: &str) -> StatSnapshot {
        registry
            .snapshot()
            .get(name)
            .unwrap_or_else(|| panic!("{name} is tracked"))
            .clone()
    }

    fn scalar(sum: i64, count: u64) -> Window {
        Window {
            sum,
            count,
            distribution: None,
        }
    }

    #[test]
    fn handles_of_one_name_share_their_data() {
        let (registry, _) = test_registry(MAX_STATS, MAX_DISTRIBUTIONS);
        // define_stats! creates a counter per thread, each through that thread's manager.
        let first = manager_of(&registry).create_counter("requests");
        let second = manager_of(&registry).create_counter("requests");
        first.increment_value(3);
        second.increment_value(4);

        let stat = read_stat(&registry, "requests");
        assert_eq!(stat.kind, StatKind::Counter);
        assert_eq!(stat.total, scalar(7, 2));
        assert_eq!(stat.recent, stat.total);
        assert_eq!(stat.total.average(), Some(3.5));
        assert_eq!(stat.total.quantile(0.5), None);
    }

    #[test]
    fn aggregated_values_count_their_samples() {
        let (registry, _) = test_registry(MAX_STATS, MAX_DISTRIBUTIONS);
        manager_of(&registry)
            .create_timeseries("bytes", &[AggregationType::Sum], &[])
            .add_value_aggregated(79, 10);

        let stat = read_stat(&registry, "bytes");
        assert_eq!(stat.kind, StatKind::Timeseries);
        assert_eq!(stat.total, scalar(79, 10));
        assert_eq!(stat.recent, scalar(79, 10));
    }

    #[test]
    fn recent_samples_leave_the_window_slot_by_slot() {
        let (registry, now) = test_registry(MAX_STATS, MAX_DISTRIBUTIONS);
        let series = manager_of(&registry).create_timeseries("t", &[], &[]);
        let window = || registry.snapshot().window;

        set_secs(&now, 3);
        assert_eq!(window(), Duration::from_secs(3), "capped by the uptime");
        set_secs(&now, 0);
        series.add_value(1);
        set_secs(&now, 30);
        series.add_value(10);
        set_secs(&now, 58);
        series.add_value(100);

        set_secs(&now, 59);
        assert_eq!(read_stat(&registry, "t").recent, scalar(111, 3));
        assert_eq!(window(), Duration::from_secs(59));

        // Slot period 12 starts: period 0, which held the first sample, has left.
        set_secs(&now, 60);
        assert_eq!(read_stat(&registry, "t").recent, scalar(110, 2));
        assert_eq!(window(), Duration::from_secs(55));
        // Period 12 reuses the slot of period 0.
        set_secs(&now, 61);
        series.add_value(1000);
        assert_eq!(read_stat(&registry, "t").recent, scalar(1110, 3));
        now.store(64_500, Ordering::SeqCst);
        assert_eq!(window(), Duration::from_millis(59_500));

        set_secs(&now, 121);
        assert_eq!(read_stat(&registry, "t").recent, scalar(0, 0));
        assert_eq!(read_stat(&registry, "t").total, scalar(1111, 4));
        assert_eq!(window(), Duration::from_secs(56));
    }

    #[test]
    fn a_late_write_keeps_a_newer_slot_period() {
        let series = Series::new(false);
        series.record(Duration::from_secs(60), 5, 1, 5);
        // Read the clock before period 12 started, and wrote after: into period 12's slot.
        series.record(Duration::from_secs(0), 7, 1, 7);
        let (total, recent) = series.read(12);
        assert_eq!(total, scalar(12, 2));
        assert_eq!(recent, scalar(12, 2));
    }

    #[test]
    fn buckets_are_contiguous_and_hold_their_values() {
        let mut values: Vec<i64> = (1..=1_000_000).collect();
        // Steps of at most 1/64th: narrower than any bucket from here on.
        let mut value: i64 = 1_000_001;
        while value < 1 << 42 {
            values.push(value);
            value += value / 64;
        }
        values.extend([(1 << 41) - 1, 1 << 41, (1 << 41) + 1, i64::MAX]);
        values.sort_unstable();
        values.dedup();

        let mut previous = 0;
        for value in values {
            let index = bucket_index(value);
            if value >= 1 << 41 {
                assert_eq!(index, BUCKETS - 1, "{value}");
                continue;
            }
            assert!(
                index == previous || index == previous + 1,
                "{value}: bucket {index} after {previous}"
            );
            let (low, high) = bucket_bounds(index);
            assert!(low <= value && value <= high, "{value} in [{low}, {high}]");
            previous = index;
        }
        assert_eq!(previous, BUCKETS - 1);
        assert_eq!(bucket_index(0), 0);
        assert_eq!(bucket_index(-5), 0);
        assert_eq!(bucket_index(i64::MAX), BUCKETS - 1);
    }

    #[test]
    fn quantiles_are_within_an_eighth() {
        let (registry, _) = test_registry(MAX_STATS, MAX_DISTRIBUTIONS);
        let manager = manager_of(&registry);
        let latency = manager.create_quantile_stat("latency", &[], &[], &[]);
        for value in 1..=1000 {
            latency.add_value(value);
        }
        let stat = read_stat(&registry, "latency");
        assert_eq!(stat.kind, StatKind::Quantile);
        assert_eq!((stat.total.sum, stat.total.count), (500_500, 1000));
        assert_eq!(stat.recent, stat.total);
        let within = |q: f64, expected: f64| {
            let actual = stat.recent.quantile(q).expect("samples") as f64;
            assert!(
                (actual - expected).abs() <= expected * 0.125,
                "p{q}: {actual} for {expected}"
            );
        };
        within(0.5, 500.0);
        within(0.99, 990.0);

        let mut small = Distribution::new();
        for value in 1..=7 {
            small.record(value, 1);
        }
        for value in 1..=7 {
            assert_eq!(small.quantile((value as f64 - 0.5) / 7.0), Some(value));
        }
        assert_eq!(small.quantile(0.0), Some(1));
        assert_eq!(Distribution::new().quantile(0.5), None);

        let repeated = manager.create_histogram(
            "repeated",
            &[],
            BucketConfig {
                width: 1,
                min: 0,
                max: 10,
            },
            &[],
        );
        repeated.add_repeated_value(5, 3);
        let stat = read_stat(&registry, "repeated");
        assert_eq!(stat.kind, StatKind::Histogram);
        assert_eq!((stat.total.sum, stat.total.count), (15, 3));
        let distribution = stat.total.distribution.expect("a histogram");
        assert_eq!(distribution.count(), 3);
        assert_eq!(distribution.quantile(1.0), Some(5));
    }

    #[test]
    fn merged_distributions_equal_one_recording() {
        let mut low = Distribution::new();
        let mut high = Distribution::new();
        let mut all = Distribution::new();
        for value in -3..5000 {
            let half = if value < 1000 { &mut low } else { &mut high };
            half.record(value, 2);
            all.record(value, 2);
        }
        assert_ne!(low, all);
        low.merge(&high);
        assert_eq!(low, all);
        assert_eq!(low.count(), 2 * 5003);
    }

    #[fbinit::test]
    fn singletons_are_unset_until_written(fb: FacebookInit) {
        let (registry, _) = test_registry(MAX_STATS, MAX_DISTRIBUTIONS);
        let factory = MemoryStatsFactory {
            registry: Arc::clone(&registry),
        };
        let first = factory.create_singleton_counter("gauge").expect("tracked");
        let second = factory.create_singleton_counter("gauge").expect("tracked");
        assert_eq!(first.get_value(fb), None);
        assert_eq!(read_stat(&registry, "gauge").total, scalar(0, 0));

        first.set_value(fb, 40);
        second.increment_value(fb, 2);
        assert_eq!(first.get_value(fb), Some(42));
        assert_eq!(second.get_value(fb), Some(42));
        let stat = read_stat(&registry, "gauge");
        assert_eq!(stat.kind, StatKind::Singleton);
        assert_eq!(stat.total, scalar(42, 1));
        assert_eq!(stat.recent, stat.total);
    }

    #[test]
    fn stats_over_a_limit_or_of_another_class_are_dropped() {
        let (registry, _) = test_registry(2, MAX_DISTRIBUTIONS);
        let manager = manager_of(&registry);
        let first = manager.create_counter("first");
        manager.create_timeseries("second", &[], &[]);
        let over = manager.create_counter("over");
        over.increment_value(1);
        first.increment_value(5);
        let snapshot = registry.snapshot();
        assert_eq!(snapshot.dropped, 1);
        assert!(snapshot.get("over").is_none());
        assert_eq!(snapshot.get("first").expect("tracked").total, scalar(5, 1));

        let (registry, _) = test_registry(MAX_STATS, 1);
        let manager = manager_of(&registry);
        manager.create_quantile_stat("q1", &[], &[], &[]);
        manager
            .create_histogram(
                "h2",
                &[],
                BucketConfig {
                    width: 1,
                    min: 0,
                    max: 10,
                },
                &[],
            )
            .add_value(1);
        manager.create_counter("scalar").increment_value(1);
        let snapshot = registry.snapshot();
        assert_eq!(snapshot.dropped, 1);
        assert!(snapshot.get("h2").is_none());
        assert!(snapshot.get("scalar").is_some());

        // A kind of the same class shares the entry; another class does not.
        let (registry, _) = test_registry(MAX_STATS, MAX_DISTRIBUTIONS);
        let manager = manager_of(&registry);
        manager.create_timeseries("t", &[], &[]).add_value(1);
        manager.create_counter("t").increment_value(2);
        manager
            .create_quantile_stat("t", &[], &[], &[])
            .add_value(4);
        let snapshot = registry.snapshot();
        assert_eq!(snapshot.dropped, 1);
        let stat = snapshot.get("t").expect("tracked");
        assert_eq!(stat.kind, StatKind::Timeseries);
        assert_eq!(stat.total, scalar(3, 2));
    }

    #[fbinit::test]
    fn singletons_count_towards_the_limit(fb: FacebookInit) {
        let (registry, _) = test_registry(1, MAX_DISTRIBUTIONS);
        let factory = MemoryStatsFactory {
            registry: Arc::clone(&registry),
        };
        manager_of(&registry).create_counter("first");
        let over = factory.create_singleton_counter("over").expect("inert");
        over.set_value(fb, 1);
        assert_eq!(over.get_value(fb), None);
        assert_eq!(registry.snapshot().dropped, 1);
    }

    #[test]
    fn snapshots_are_sorted_by_name() {
        let (registry, _) = test_registry(MAX_STATS, MAX_DISTRIBUTIONS);
        let manager = manager_of(&registry);
        let names = ["b.c", "a", "b", "a.b", "c", "B", "a_b"];
        for name in names {
            manager.create_counter(name).increment_value(1);
        }
        let snapshot = registry.snapshot();
        let mut sorted = names.map(String::from).to_vec();
        sorted.sort();
        let listed: Vec<String> = snapshot
            .stats
            .iter()
            .map(|stat| stat.name.clone())
            .collect();
        assert_eq!(listed, sorted);
        for name in names {
            assert_eq!(snapshot.get(name).expect("listed").name, name);
        }
        assert!(snapshot.get("d").is_none());
    }

    #[test]
    fn concurrent_writes_are_all_counted() {
        let (registry, _) = test_registry(MAX_STATS, MAX_DISTRIBUTIONS);
        thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    let counter = manager_of(&registry).create_counter("hot");
                    for _ in 0..100_000 {
                        counter.increment_value(1);
                    }
                });
            }
        });
        assert_eq!(read_stat(&registry, "hot").total, scalar(800_000, 800_000));
    }
}
