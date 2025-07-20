// codeforces110a.cpp
#include <iostream>
using namespace std;
bool islucky(long long int n) {
  bool t = n;
  while (n > 0) {
    if (n % 10 != 4 && n % 10 != 7) {
      t = false;
      break;
    }
    n = n / 10;
  }
  return t;
}
int main(int argc, char const *argv[]) {
  long long int n, b;
  long long int d = 0;

  cin >> n;
  b = n;
  while (b) {
    if (b % 10 == 4 || b % 10 == 7)
      d++;
    b /= 10;
  }
  if (islucky(d))
    cout << "YES";
  else
    cout << "NO";
  return 0;
}