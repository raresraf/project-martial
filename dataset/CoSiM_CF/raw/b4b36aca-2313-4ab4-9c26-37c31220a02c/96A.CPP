#include <cstdlib>
#include <iostream>
using namespace std;
int main() {

  string txt;
  cin >> txt;
  int i = 0, db = 0;
  char akt = '0';

  while (i < txt.length() && db != 7) {
    txt[i] == akt ? db++ : db = 1, akt = txt[i];
    i++;
  }

  db == 7 ? cout << "YES" : cout << "NO";

  return 0;
}
