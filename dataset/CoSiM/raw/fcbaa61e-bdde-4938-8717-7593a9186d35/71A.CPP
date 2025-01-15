#include <bits/stdc++.h>
using namespace std;

void change() {
  string s;
  cin >> s;
  int size = s.size();
  if (size > 10) {
    cout << s[0] << size - 2 << s[size - 1] << endl;
  } else {
    cout << s << endl;
  }
}

int main() {
  int n;
  cin >> n;
  while (n--) {
    change();
  }
}
