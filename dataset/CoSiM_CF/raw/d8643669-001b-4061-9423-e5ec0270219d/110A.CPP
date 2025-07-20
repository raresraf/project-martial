#include <iostream>
using namespace std;
int main() {
  unsigned long long n;
  int flag = 0, count = 0, c = 0;
  cin >> n;
  while (n) {
    if (n % 10 == 4 || n % 10 == 7) {
      count++;
    }
    n /= 10;
  }
  if (count == 4 || count == 7)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
