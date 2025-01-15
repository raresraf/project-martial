#include <iostream>
using namespace std;
int main() {
  int m, n;
  cin >> m >> n;
  int count = 0;
  if (m % 2 == 0) {
    count = (m / 2) * n;
  } else {
    count = (m / 2) * n + (n / 2);
  }
  cout << count << endl;
}
