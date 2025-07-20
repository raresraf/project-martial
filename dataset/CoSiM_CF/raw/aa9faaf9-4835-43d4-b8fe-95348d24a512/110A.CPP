#include <iostream>
using namespace std;
long long int cnt_lucky(long long int temp) {
  long long int cnt = 0;
  while (temp > 0) {
    if (temp % 10 == 4 || temp % 10 == 7)
      cnt++;
    temp /= 10;
  }
  return cnt;
}
bool is_lucky(long long int n) {
  if (n == 0)
    return false;
  while (n > 0) {
    if (n % 10 != 7 && n % 10 != 4) {
      return false;
    }
    n /= 10;
  }
  return true;
}
int main() {
  // your code goes here
  long long int n;
  cin >> n;
  long long int temp = cnt_lucky(n);
  if (is_lucky(temp)) {
    cout << "YES\n";
  } else {
    cout << "NO\n";
  }
  return 0;
}