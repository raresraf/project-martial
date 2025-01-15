#include <iostream>
#include <set>
#define optimizar_io                                                           \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);
using namespace std;
string a;
set<char> lol;
int main() {
  optimizar_io cin >> a;
  for (int i = 0; i < (int)a.size(); ++i)
    lol.insert(a[i]);
  cout << ((int)lol.size() % 2 == 0 ? "CHAT WITH HER!\n" : "IGNORE HIM!\n");
  return 0;
}
