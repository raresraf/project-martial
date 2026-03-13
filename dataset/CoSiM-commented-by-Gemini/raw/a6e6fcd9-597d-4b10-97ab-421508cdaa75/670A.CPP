/**
 * @file 670A.CPP
 * @brief Solution for CodeForces Problem 670A - "Holidays".
 *
 * @details
 * This program calculates the minimum and maximum possible number of weekend days (days off)
 * that can occur within a period of `n` consecutive days. A week has 5 workdays and 2 weekend days.
 *
 * @algorithm Simulation
 * - To find the minimum number of days off, we should start the n-day period on a Monday.
 *   This arrangement maximizes the number of weekdays at the beginning of the period.
 * - To find the maximum number of days off, we should start the n-day period on a Saturday.
 *   This arrangement maximizes the number of weekend days at the beginning.
 *
 * The code simulates this by representing a week with an array where 1 indicates a weekend day.
 * - `a[] = {0, 0, 0, 0, 0, 1, 1}` corresponds to {Mon, Tue, Wed, Thu, Fri, Sat, Sun}.
 * - The minimum is found by starting the simulation loop with an index corresponding to Monday (`t=0`).
 * - The maximum is found by starting with an index corresponding to Saturday (`t=5`).
 *
 * An analytical solution is also possible:
 * - Min days off: `(n / 7) * 2 + max(0, (n % 7) - 5)`
 * - Max days off: `(n / 7) * 2 + min(2, n % 7)`
 *
 * @time_complexity O(N), where N is the number of days, due to the simulation loop.
 * @space_complexity O(1).
 *
 * @note The file includes many preprocessor directives and type definitions common in
 * competitive programming templates, which are not used in this specific solution.
 */
#include <bits/stdc++.h>
// Other includes and defines from original file are omitted for brevity as they are unused.

using namespace std;

#define ll long long

ll n; // The total number of days in the period.

void Input() {
    cin >> n;
}

void Solve() {
    // Array representing the week: 0 for workday, 1 for weekend.
    // Index 0=Mon, 1=Tue, ..., 5=Sat, 6=Sun.
    ll a[] = {0, 0, 0, 0, 0, 1, 1};

    // --- Calculate Minimum Days Off ---
    // We assume the period starts on a Monday to minimize weekend days.
    ll t = 0; // Start on Monday (index 0).
    ll min_ans = 0;
    for (ll i = 0; i < n; i++) {
        if (a[t])
            min_ans++;
        t = (t + 1) % 7; // Go to the next day of the week.
    }
    cout << min_ans << " ";

    // --- Calculate Maximum Days Off ---
    // We assume the period starts on a Saturday to maximize weekend days.
    t = 5; // Start on Saturday (index 5).
    ll max_ans = 0;
    for (ll i = 0; i < n; i++) {
        if (a[t])
            max_ans++;
        t = (t + 1) % 7; // Go to the next day of the week.
    }
    cout << max_ans;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
#ifndef ONLINE_JUDGE
    // This block is for local testing, redirecting I/O to files.
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif

    Input();
    Solve();

    return 0;
}
