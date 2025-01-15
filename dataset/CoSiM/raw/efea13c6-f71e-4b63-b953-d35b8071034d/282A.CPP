#include <iostream>

using namespace std;

int main() {
  int n;
  int res = 0;
  char a, b, c;
  cin >> n;
  while (n--) {
    cin >> a >> b >> c;
    if (a == '-' || b == '-' || c == '-') {
      res--;
    } else {
      res++;
    }
  }
  cout << res << endl;
  return 0;
}