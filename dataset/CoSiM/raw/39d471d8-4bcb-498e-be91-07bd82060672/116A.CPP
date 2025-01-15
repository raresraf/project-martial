#include <iostream>
using namespace std;
#define soads_game int main()

soads_game {
  int n, t = 0, m = 0, d = 0, a = 0;
  cin >> n;
  while (n--) {
    cin >> a >> d;
    t = t - a + d;
    m = max(m, t);
  }
  cout << m;
  return 0;
}
