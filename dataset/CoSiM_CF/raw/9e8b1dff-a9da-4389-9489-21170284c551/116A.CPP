#include <iostream>

using namespace std;

int main() {
  int n, i;
  cin >> n;

  int a, b;
  int capacity = 0;
  int ans = 0;

  for (i = 0; i < n; i++) {
    cin >> a >> b;

    if (i == 0)
      capacity = b + a;
    else {
      capacity = capacity - a + b;
    }

    if (capacity > ans)
      ans = capacity;
  }

  cout << ans;

  return 0;
}
