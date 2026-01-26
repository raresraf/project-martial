#include <bits/stdc++.h>
using namespace std;
int main(int argc, char const *argv[]) {
  /* code */
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);

  int x = 0;
  int t;
  cin >> t;
  while (t)

  {
    string str;
    cin >> str;

    if (str == "X++" || str == "++X") {
      x = x + 1;
    }

    if (str == "X--" || str == "--X") {
      x = x + 1;
    }
    t--;
  }
  cout << x;
  return 0;
}
