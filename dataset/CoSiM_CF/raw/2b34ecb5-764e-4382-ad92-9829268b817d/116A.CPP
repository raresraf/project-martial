#include <iostream>
using namespace std;

int main() {

  int n;
  cin >> n;
  int a, b;
  int current = 0;
  int min = 0;
  while (n--) {
    cin >> a >> b;
    current = current - a + b;
    if (current > min)
      min = current;
  }

  cout << min;

  return 0;
}