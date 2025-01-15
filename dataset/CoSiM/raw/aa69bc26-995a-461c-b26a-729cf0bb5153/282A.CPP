#include <iostream>
using namespace std;

int main(void) {
  int n;
  cin >> n;

  int ans = 0;

  while (n--) {
    string op;
    cin >> op;

    if (op == "X++" || op == "++X")
      ans++;
    else
      ans--;
  }

  cout << ans << endl;

  return 0;
}