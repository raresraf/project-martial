#include "bits/stdc++.h"
#define do_not_sync                                                            \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);
#define FIO                                                                    \
  freopen("input.txt", "r", stdin);                                            \
  freopen("output.txt", "w", stdout);
#define w(t)                                                                   \
  int t;                                                                       \
  cin >> t;                                                                    \
  while (t--)

using namespace std;

int main(void) {
  do_not_sync
      // FIO
      int n,
      t;
  cin >> n >> t;
  string s;
  cin >> s;

  while (t--) {
    for (int i = 0; i < n - 1; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        i++;
      }
    }
  }

  cout << s << endl;

  return 0;
}
