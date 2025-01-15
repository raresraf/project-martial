#include <bits/stdc++.h>
using namespace std;
int main() {
  // freopen("input.txt", "r", stdin);
  char ch;
  string s;
  scanf("%c", &ch);
  cin >> s;
  if (ch >= 'a' && ch <= 'z')
    printf("%c", ch - 'a' + 'A');
  else
    printf("%c", ch);
  cout << s << endl;
  return 0;
}
