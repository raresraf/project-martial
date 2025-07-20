#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;

  int total = 0;
  while (n--) {
    int p, v, t;
    cin >> p >> v >> t;
    if (p + v + t >= 2)
      total++;
  }
  cout << total;
}