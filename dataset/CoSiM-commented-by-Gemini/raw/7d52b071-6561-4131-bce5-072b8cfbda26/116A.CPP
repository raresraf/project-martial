/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A, "Tram".
 * @details This file contains a solution to the "Tram" problem, which involves
 * calculating the minimum required capacity of a tram based on the number of
 * passengers entering and exiting at several stops.
 *
 * Note: The file contains a large block of commented-out code at the top,
 * which appears to be unrelated to the active solution and is ignored.
 *
 * Algorithm (Active Code):
 * 1. Read the total number of stops.
 * 2. Initialize `sum` to track the current number of passengers and `ans` to
 *    track the maximum number of passengers seen so far (the minimum capacity).
 * 3. Process the first stop's data to initialize the passenger count.
 * 4. Loop through the remaining stops.
 * 5. For each stop, update the current passenger count by subtracting the
 *    number of exiting passengers and adding the number of entering passengers.
 * 6. After each update, compare the current passenger count with the maximum
 *    seen so far and update the maximum if necessary.
 * 7. After all stops are processed, print the final maximum value.
 *
 * Time Complexity: O(N), where N is the number of stops.
 * Space Complexity: O(1).
 */
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
#include <bits/stdc++.h>
#define elif else if
#define pb push_back
#define mp make_pair
#define f first
#define second
typedef long long int ll;
#define rep(i, a, b) for (ll i = a; i < b; i++)
using namespace std;

/**
 * @brief Contains the core logic for solving the Tram problem.
 */
void solve() {
  ll count = 0; // Unused variable.
  ll sum = 0;   // Tracks the current number of passengers on the tram.
  ll query;
  cin >> query; // Number of tram stops.
  ll res = query - 1;
  ll a, b;
  cin >> a >> b; // Read exits (a) and entrances (b) for the first stop.
  
  ll ans = INT_MIN; // Stores the maximum passenger count found so far.

  // Initialize passenger count. Since the tram starts empty, a=0, so sum becomes b.
  sum += a + b;
  ans = max(ans, sum);

  /**
   * @brief Loop through the remaining stops to simulate passenger flow.
   */
  while (res--) {
    ll p, q;
    cin >> p >> q; // Read exits (p) and entrances (q).
    sum -= p;      // Passengers exit.
    sum += q;      // Passengers enter.
    ans = max(ans, sum); // Update the required capacity if current count is higher.
  }
  cout << ans << endl;
}

/**
 * @brief Main function, entry point of the program.
 */
int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);
  ll t = 1;
  // cin>>t; // The code is set up for multiple test cases but runs only one.
  while (t--) {
    solve();
  }
  return 0;
}