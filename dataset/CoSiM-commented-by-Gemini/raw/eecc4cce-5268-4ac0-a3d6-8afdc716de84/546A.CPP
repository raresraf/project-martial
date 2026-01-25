#include <iostream>
using namespace std;

int main() {
  int w, k, n, i;
  int total = 0;
  int pay;
  cin >> k >> n >> w;
  for (i = 1; i <= w; i++) {
    total = k * i + total;
  }
  cout << total << endl;
  if (n > total) {
    cout << "0";
  } else {
    pay = total - n;
    cout << pay;
  }

  return 0;
}
