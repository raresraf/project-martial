#include <bits/stdc++.h>
#define ll long long
#define pb push_back
#define mp make_pair
#define fr(i, j) for (i = 0; i < j; i++)

using namespace std;

int main() {
  string str;
  string temp1 = "", res = "";
  int i = 0;
  char c;
  cin >> str;
  int n = str.length();

  fr(i, n) {
    c = str[i];

    if (c >= 'A' && c <= 'Z')
      str[i] = c + ('a' - 'A');

    c = str[i];
    if (c != 'a' && c != 'e' && c != 'i' && c != 'o' && c != 'u' && c != 'y')
      temp1 += c;
  }

  n = temp1.length();
  fr(i, n) { res = res + "." + temp1[i]; }

  cout << res;
}
