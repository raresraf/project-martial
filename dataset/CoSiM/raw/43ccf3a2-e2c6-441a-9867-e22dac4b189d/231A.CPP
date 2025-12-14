#include <iostream>
using namespace std;
int main() {
  int n, counter = 0, a, b, c;
  cin >> n;
  while (n--) {
    cin >> a >> b >> c;
    if (a + b + c > 1) {
      counter++;
    }
  }
  cout << counter << endl;
}
