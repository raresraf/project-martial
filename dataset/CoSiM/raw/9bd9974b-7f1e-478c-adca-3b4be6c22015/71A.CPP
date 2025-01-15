#include <bits/stdc++.h>

using namespace std;

string to_str(int i) {

  stringstream ss;

  ss << i;

  string s;

  ss >> s;

  return s;
}

int main() {

  int n;

  string s, ans;

  scanf("%d", &n);

  while (n--) {

    cin >> s;

    ans = "";

    int len = s.length();

    if (len > 10)
      ans += s[0] + to_str(len - 2) + s[len - 1];
    else
      ans = s;

    cout << ans << endl;
  }
}