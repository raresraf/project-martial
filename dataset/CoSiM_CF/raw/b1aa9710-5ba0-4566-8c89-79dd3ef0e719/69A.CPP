#include <iostream>
using namespace std;
int main() {
  long long r1 = 0, r2 = 0, r3 = 0;
  int n, num1, num2, num3;
  cin >> n;
  for (int i = 1; i <= n; i++) {
    cin >> num1 >> num2 >> num3;
    r1 += num1;
    r2 += num2;
    r3 += num3;
  }
  if (r1 == 0 && r2 == 0 && r3 == 0) {
    cout << "YES";
  } else
    cout << "NO";
  return 0;
}
