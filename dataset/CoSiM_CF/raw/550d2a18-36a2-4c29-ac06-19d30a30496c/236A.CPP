#include <iostream>
#include <map>
#include <string>
using namespace std;
void solved() {
  string s;
  cin >> s;
  map<char, bool> mapp;
  int dem = 0;
  for (int i = 0; i < s.length(); i++) {
    if (mapp[s[i]] == false) {
      mapp[s[i]] = true;
      dem++;
    }
  }
  if (dem % 2 != 0)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";
}
int main() {
  solved();
  return 0;
}
