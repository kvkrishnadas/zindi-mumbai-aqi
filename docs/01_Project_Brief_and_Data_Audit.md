Test.csv has 2,592 rows (4 stations × 648 hours) and is NOT sorted chronologically—row 1 starts at 2026-10-27T01:00+05:30.
Train.csv['timestamp_ist'] marks the end of the hour, matching Test.csv['target_end_ist'] (not target_start_ist), and mean is a duplicate of pm25.   
Private scoring is strictly 192 rows (2026-11-11T01:00 to 2026-11-13T00:00), meaning 100% of final prize ranking depends on Day +3 and Day +4 post-Diwali.   
Hardware sentinel spikes (99999.0, 75203.5, flag_high_1000 == 1) last up to 4 hours and are followed by 0.0000 post-reboot dropouts.   
Rolling updates introduced a 5-day withheld hole (Oct 24–28) inside Train.csv.   
AdditionalHistory.zip triage: keep Other_Mumbai_Stations_2025_2026.csv.gz (9,125 rows) for cross-city reference, filter History_Mumbai_2024.csv.gz (216,472 rows) on exact station_name and twin site IDs (site_5113, site_5102, site_5960, site_5961) with Median-IQR alignment to fix the ~23 µg/m³ offset, and drop Legacy_Observations_2020_2022.csv.gz completely.   
Git Checkpoint & Chat Close: In VS Code Source Control, commit docs: add forensic EDA audit findings and click Sync Changes. Close the Spoke 1 Chat permanently.