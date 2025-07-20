#include <bits/stdc++.h>
using namespace std;

int main() {
  long long n;
  int counter = 0;
  cin >> n;

  while (n != 0) {
    if (n % 10 == 7 || n % 10 == 4) {
      counter++;
    }
    n = n / 10;
  }

  if (counter == 7) {
    cout << "YES";
  } else if (counter == 4) {
    cout << "YES";
  } else {
    cout << "NO";
  }

  return 0;
}
