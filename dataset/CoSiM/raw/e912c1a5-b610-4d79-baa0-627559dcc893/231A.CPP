#include <iostream>
using namespace std;
int main(int argc, char const *argv[]) {
  int t, count, a, b, c;
  cin >> t;
  count = 0;
  while (t--) {
    cin >> a >> b >> c;
    if (a + b + c >= 2) {
      count++;
    }
  }
  cout << count;

  return 0;
}
