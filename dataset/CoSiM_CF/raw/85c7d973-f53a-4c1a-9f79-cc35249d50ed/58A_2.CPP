#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  int a = s.find('h');
  int b = s.find('e', a);
  int c = s.find('l', b);
  int d = s.find('l', c + 1);
  int e = s.find('o', d);

  // cout<<a<<"\t"<<b<<' '<<c<<' '<<d<<" "<<e<<"\n";
  if (a != -1 && b != -1 && c != -1 && d != -1 && e != -1)
    cout << "YES\n";
  else
    cout << "NO\n";
  return 0;
}
