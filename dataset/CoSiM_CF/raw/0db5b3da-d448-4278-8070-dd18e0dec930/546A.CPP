#include <iostream>

using std::cin;
using std::cout;
using std::endl;
int cost(int, int);

int main() {
  int k, n, w;
  cin >> k >> n >> w;
  int borrowNeeded = cost(k, w) - n;

  if (borrowNeeded < 0) {
    cout << "0";
  } else {
    cout << borrowNeeded;
  }

  return 0;
}

int cost(int k, int i) {
  if (i == 1) {
    return k;
  }

  return i * k + cost(k, i - 1);
}
