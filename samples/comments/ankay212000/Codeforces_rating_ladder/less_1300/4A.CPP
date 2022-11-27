#include <bits/stdc++.h>
#define ll long long
#define io                                                                     \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL)
using namespace std;
#define N 1000000000000
#define mod 1073741824

int main() {
  io;
  ll test_case;
  // cin>>test_case;
  test_case = 1;
  while (test_case--) {
    ll n, temp, count1(0), count0(0), maxlen = -1;
    cin >> n;
    for (int i = 0; i < n; i++) {
      cin >> temp;
      if (temp == 1) {
        count1++;
        if (count0 > 0)
          count0--;
      } else {
        count0++;
        if (count0 > maxlen)
          maxlen = count0;
      }
    }

    cout << count1 + maxlen;
  }
  return 0;
}