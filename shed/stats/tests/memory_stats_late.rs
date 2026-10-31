/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is dual-licensed under either the MIT license found in the
 * LICENSE-MIT file in the root directory of this source tree or the Apache
 * License, Version 2.0 found in the LICENSE-APACHE file in the root directory
 * of this source tree. You may select, at your option, one of the
 * above-listed licenses.
 */

//! Installing the in-memory backend after a stat was used. Its own process, as it leaves the
//! default factory registered.

use stats::memory_stats;
use stats::prelude::*;

define_stats! {
    prefix = "memory_stats_late_test";
    early: timeseries(Sum),
}

#[test]
fn install_fails_once_a_stat_was_used() {
    STATS::early.add_value(1);
    assert!(!memory_stats::install());
    assert!(!memory_stats::install());
    assert!(!memory_stats::is_installed());
    assert!(memory_stats::snapshot().is_none());
}
