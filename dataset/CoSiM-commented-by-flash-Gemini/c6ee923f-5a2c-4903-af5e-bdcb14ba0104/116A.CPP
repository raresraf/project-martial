#include <iostream>
using namespace std;
int main() {
  int n, a, b;
  int count = 0, max;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    count -= a;
    count += b;
    if (i == 0) {
      max = count;
    }

    if (max < count)
      max = count;
  }
  cout << max;
  return 0;
}
