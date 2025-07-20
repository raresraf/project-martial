#include <iostream>

using namespace std;

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);

  int n;
  cin >> n;

  int max_p = -1;
  int p = 0;

  for (int i = 0; i < n; i++) {
    int a, b;
    cin >> a >> b;

    if (i == 0) {
      p += b;
      max_p = max(p, max_p);
    } else {
      p -= a;
      p += b;
      max_p = max(p, max_p);
    }
  }

  cout << max_p << endl;

  return 0;
}
