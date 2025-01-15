#include <iostream>
#include <set>

using namespace std;
int main() {

  string str;
  set<char> ans;
  cin >> str;

  for (char x : str) {
    if (ans.count(x) == 0)
      ans.insert(x);
  }
  int len = ans.size();

  if (len % 2 == 1)
    cout << "IGNORE HIM!" << endl;
  else
    cout << "CHAT WITH HER!" << endl;

  return 0;
}
