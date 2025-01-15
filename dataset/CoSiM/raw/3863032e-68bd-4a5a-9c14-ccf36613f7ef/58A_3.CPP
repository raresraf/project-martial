#include <iostream>
using namespace std;
#define soads_game int main()

soads_game {
  int n, t = 1, i = 0, m = 1;
  cin >> n;
  int a[n];
  while (n--) {
    cin >> a[i];
    if (i > 0) {
      if (a[i - 1] <= a[i]) {
        t++;
        m = max(t, m);
      } else {
        t = 1;
      }
    }

    ;
    i++;
  }
  cout << m;
  return 0;
}
