#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  int n, sum = 0, max1 = 0;

  cin >> n;

  while (n--) {
    int a, b;
    cin >> a >> b;
    sum += b - a;
    max1 = max(sum, max1);
  }
  cout << max1 << endl;
  return 0;
}
