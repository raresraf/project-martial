#include <bits/stdc++.h>
using namespace std;

int main(void) {
  int lucky = 0;
  string num;

  cin >> num;

  for (auto i = num.begin(); i != num.end(); ++i) {
    if (*i == '4' || *i == '7') {
      ++lucky;
    }
  }

  if (lucky == 4 || lucky == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }

  return 0;
}
