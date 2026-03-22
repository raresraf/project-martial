/*#include<bits/stdc++.h>
typedef long long int ll;
const ll maxy=1e5+5;
#define rep(i,a,b) for(ll i=a;i<b;i++)
using namespace std;
ll sz[maxy];
void fill(){
    memset(sz,0,sizeof(sz));
}
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    ll pre;bool ok=0;
    cin>>pre;
    ll p=pre-1;while(p--){
        ll res;cin>>res;sz[res]++;
    } ll ans=0;
    rep(i,0,maxy){
        if(sz[i]>0 and sz[i]>=3){
            continue;
        }else if(sz[i]>0 and sz[i]<=2){ok=1;break;}
    }if(ok){cout<<"No"<<endl;}else{cout<<"Yes"<<endl;}
    return 0;
}*/
/**
 * @file 116A.CPP
 * @brief Solution to Codeforces problem 116A - Tram.
 *        This program simulates passengers entering and exiting a tram,
 *        calculating the maximum number of passengers simultaneously present in the tram at any point.
 *        The initial code in the commented-out block seems to be for a different problem
 *        or an earlier approach, but the active code solves the Tram problem efficiently.
 *
 * Algorithm (Active Code):
 *   1. Reads an integer `query` (likely representing the number of stops/events).
 *   2. Reads initial number of passengers exiting (`a`) and entering (`b`) at the first stop.
 *      Initializes `sum` with `a + b` (representing passengers after first event).
 *   3. Initializes `ans` (maximum passengers) to `INT_MIN`.
 *   4. Updates `ans` with the current `sum`.
 *   5. Loops `query - 1` times for subsequent stops:
 *      a. Reads passengers exiting (`p`) and entering (`q`) at the current stop.
 *      b. Updates `sum` by subtracting `p` and adding `q`.
 *      c. Updates `ans` with the maximum of its current value and the new `sum`.
 *   6. Prints the final `ans`.
 *
 * Time Complexity: O(query), where `query` is the number of events/stops, due to the single loop.
 * Space Complexity: O(1) for storing a few integer variables.
 */
#include <bits/stdc++.h> // Includes common C++ standard library headers.
#define elif else if     // Macro for 'else if'.
#define pb push_back     // Macro for `vector::push_back`.
#define mp make_pair     // Macro for `std::make_pair`.
#define f first          // Macro for `std::pair::first`.
#define second           // Incomplete macro, likely intended for `std::pair::second`.
typedef long long int ll; // Typedef for long long int.
#define rep(i, a, b) for (ll i = a; i < b; i++) // Macro for a for-loop.
using namespace std;

/**
 * @brief Solves the "Tram" problem.
 *
 * Tracks the number of passengers in a tram over several stops and determines
 * the maximum capacity needed.
 */
void solve() {
  ll count = 0; // Declared but unused in this specific solution.
  ll sum = 0;   // Represents the current number of passengers in the tram.
  ll query;     // Represents the number of stops or events.
  cin >> query;
  ll res = query - 1; // Remaining events to process after the first.
  ll a, b;            // Passengers exiting (`a`) and entering (`b`) at the first event.
  cin >> a >> b;
  ll ans = INT_MIN; // Stores the maximum number of passengers seen. Initialized to minimum possible integer.

  // Initial state: passengers after the first event.
  sum += a + b;
  // Update the maximum passengers with the initial sum.
  ans = max(ans, sum);

  /**
   * Block Logic: Iterates through the remaining `query - 1` events (stops).
   * Invariant: `sum` accurately reflects the number of passengers after each event.
   * `ans` stores the maximum `sum` encountered so far.
   * Pre-condition: `query`, `a`, `b` have been read, `sum` and `ans` are initialized.
   */
  while (res--) {
    ll p, q; // Passengers exiting (`p`) and entering (`q`) at the current event.
    cin >> p >> q;
    sum -= p; // Passengers exit the tram.
    sum += q; // Passengers enter the tram.
    // Update the maximum passengers encountered.
    ans = max(ans, sum);
  }
  // Print the maximum number of passengers that were ever in the tram.
  cout << ans << endl;
}

int main() {
  /**
   * @brief Main function.
   *
   * Sets up fast I/O and calls the `solve` function to execute the problem logic.
   * It is structured to handle multiple test cases, though `t` is fixed to 1 here.
   *
   * @return int Returns 0 upon successful execution.
   */
  ios::sync_with_stdio(false); // Disables synchronization with C's stdio.
  cin.tie(nullptr);            // Unties cin from cout, speeding up I/O.
  cout.tie(nullptr);           // Unties cout, ensuring immediate output.
  ll t = 1;                    // Number of test cases, currently fixed to 1.
  // cin>>t; // Can be uncommented to read number of test cases from input.
  /**
   * Block Logic: Loop for handling multiple test cases.
   * Invariant: `t` decrements with each iteration, `solve()` is called once per test case.
   */
  while (t--) {
    solve(); // Calls the function implementing the problem's logic.
  }
  return 0;
}