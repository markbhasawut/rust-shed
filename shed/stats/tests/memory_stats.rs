/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is dual-licensed under either the MIT license found in the
 * LICENSE-MIT file in the root directory of this source tree or the Apache
 * License, Version 2.0 found in the LICENSE-APACHE file in the root directory
 * of this source tree. You may select, at your option, one of the
 * above-listed licenses.
 */

//! The in-memory backend as this process's stats manager factory. A single test, so that
//! nothing uses a stat before `install`.

use std::thread;

use fbinit::FacebookInit;
use stats::memory_stats;
use stats::memory_stats::StatKind;
use stats::prelude::*;
use stats_traits::stat_types::SingletonCounter;
use stats_traits::stats_manager::BoxStatsManager;
use stats_traits::stats_manager::StatsManagerFactory;

define_stats! {
    prefix = "memory_stats_test";
    c: counter(),
    t: timeseries(Sum, Count),
    h: histogram(1, 0, 1000, Average; P 50),
    q: quantile_stat(Count; P 50, P 99; Duration::from_secs(60)),
    s: singleton_counter(),
    dc: dynamic_counter("dc.{}", (key: &'static str)),
    dt: dynamic_timeseries("dt.{}", (key: &'static str); Sum),
    dh: dynamic_histogram("dh.{}", (key: &'static str); 1, 0, 1000, Average; P 50),
    dq: dynamic_quantile_stat("dq.{}", (key: &'static str); Count; P 50; Duration::from_secs(60)),
    ds: dynamic_singleton_counter("ds.{}", (key: &'static str)),
}

define_stats_struct! {
    StructStats("memory_stats_test.struct.{}", key: &'static str),
    c: counter(),
    t: timeseries(Sum),
    h: histogram(1, 0, 1000, Average; P 50),
    q: quantile_stat(Count; P 50; Duration::from_secs(60)),
    s: singleton_counter(),
}

struct OtherFactory;

impl StatsManagerFactory for OtherFactory {
    fn create(&self) -> BoxStatsManager {
        unreachable!("never registered")
    }
}

fn write(fb: FacebookInit, struct_stats: &StructStats) {
    STATS::c.increment_value(1);
    STATS::t.add_value(5);
    STATS::h.add_value(7);
    STATS::q.add_value(6);
    STATS::s.increment_value(fb, 21);
    STATS::dc.increment_value(3, ("a",));
    STATS::dt.add_value(4, ("a",));
    STATS::dh.add_value(5, ("a",));
    STATS::dq.add_value(4, ("a",));
    STATS::ds.increment_value(fb, 50, ("a",));
    struct_stats.c.increment_value(1);
    struct_stats.t.add_value(6);
    struct_stats.h.add_value(3);
    struct_stats.q.add_value(2);
    struct_stats.s.increment_value(fb, 30);
}

#[fbinit::test]
fn the_installed_backend_tracks_every_kind_of_stat(fb: FacebookInit) {
    assert!(!memory_stats::is_installed());
    assert!(memory_stats::snapshot().is_none());
    assert!(memory_stats::install());
    assert!(memory_stats::install(), "install is idempotent");
    assert!(memory_stats::is_installed());
    assert!(!stats::try_register_stats_manager_factory(OtherFactory));

    let struct_stats = StructStats::new("key");
    write(fb, &struct_stats);
    thread::scope(|scope| {
        scope.spawn(|| write(fb, &struct_stats));
    });

    let snapshot = memory_stats::snapshot().expect("installed");
    assert_eq!(snapshot.dropped, 0);
    // Distribution samples are at most 7, so their quantiles are exact.
    let expected = [
        ("memory_stats_test.c", StatKind::Counter, 2, 2),
        ("memory_stats_test.t", StatKind::Timeseries, 10, 2),
        ("memory_stats_test.h", StatKind::Histogram, 14, 2),
        ("memory_stats_test.q", StatKind::Quantile, 12, 2),
        ("memory_stats_test.s", StatKind::Singleton, 42, 1),
        ("memory_stats_test.dc.a", StatKind::Counter, 6, 2),
        ("memory_stats_test.dt.a", StatKind::Timeseries, 8, 2),
        ("memory_stats_test.dh.a", StatKind::Histogram, 10, 2),
        ("memory_stats_test.dq.a", StatKind::Quantile, 8, 2),
        ("memory_stats_test.ds.a", StatKind::Singleton, 100, 1),
        ("memory_stats_test.struct.key.c", StatKind::Counter, 2, 2),
        (
            "memory_stats_test.struct.key.t",
            StatKind::Timeseries,
            12,
            2,
        ),
        ("memory_stats_test.struct.key.h", StatKind::Histogram, 6, 2),
        ("memory_stats_test.struct.key.q", StatKind::Quantile, 4, 2),
        ("memory_stats_test.struct.key.s", StatKind::Singleton, 60, 1),
    ];
    for (name, kind, sum, count) in expected {
        let stat = snapshot
            .get(name)
            .unwrap_or_else(|| panic!("{name} in {:?}", snapshot.stats));
        assert_eq!(stat.kind, kind, "{name}");
        assert_eq!((stat.total.sum, stat.total.count), (sum, count), "{name}");
        assert_eq!(stat.recent, stat.total, "{name}");
        let distribution = matches!(kind, StatKind::Histogram | StatKind::Quantile);
        assert_eq!(
            stat.total.quantile(0.5),
            distribution.then_some(sum / 2),
            "{name}"
        );
    }
    assert_eq!(snapshot.stats.len(), expected.len());
    assert_eq!(STATS::s.get_value(fb), Some(42));
}
