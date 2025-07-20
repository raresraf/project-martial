// 282A - Bit++

#include <iostream>

using namespace std;

int main() {
  char x[5];
  int n, sum = 0;

  cin >> n;

  while (n--) {
    cin >> x;

    if (x[1] == '+')
      sum++;
    else
      sum--;
  }

  cout << sum;

  return 0;
}
