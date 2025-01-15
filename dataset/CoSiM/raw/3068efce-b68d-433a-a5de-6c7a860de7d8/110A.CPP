#include <iostream>

using namespace std;

int main() {
  long long int n;
  int c = 0;
  cin >> n;
  while (n) {
    if (n % 10 == 7 || n % 10 == 4) {
      c++;
    }
    n /= 10;
  }
  if (c == 4 || c == 7)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
