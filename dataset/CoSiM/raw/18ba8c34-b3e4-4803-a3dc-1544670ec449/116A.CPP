#include <iostream>
using namespace std;
int main() {

  int n;
  cin >> n;

  int e, i;
  int x = 0;
  int max = 0;

  for (int j = 0; j < n; j++) {
    cin >> e;
    cin >> i;
    x -= e;
    x += i;

    if (x > max)
      max = x;
  }

  cout << max;
  return 0;
}