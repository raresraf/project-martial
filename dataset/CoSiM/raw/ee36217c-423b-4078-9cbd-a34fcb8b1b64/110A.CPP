#include <iostream>
#include <sstream>
using namespace std;
#define soads_game int main()

soads_game {
  string s;
  cin >> s;
  int con = 0;
  bool r = true;

  for (int i = 0; i < s.length(); i++) {
    if ((s[i] == '4') || (s[i] == '7'))
      con++;
  }

  stringstream gone;
  gone << con;
  string n;
  gone >> n;

  for (int j = 0; j < n.length(); j++) {
    if ((n[j] != '4') && (n[j] != '7'))
      r = false;
  }

  if (r)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
