#include <iostream>
using namespace std;

int main() {

  int n;
  int ans = 0;
  string str;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> str;
    if (str == "X++" || str == "++X")
      ans++;
    else
      ans--;
  }

  cout << ans << endl;
  return 0;
}
