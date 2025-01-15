#include <bits/stdc++.h>
using namespace std;

int main() {
  int x;
  cin >> x;

  string str;
  cin >> str;

  regex patRed(".*RR.*");
  regex patBlu(".*BB.*");
  regex patGre(".*GG.*");

  regex redRpl("RR");
  regex bluRpl("BB");
  regex greRpl("GG");

  int count = 0;

  while (regex_match(str, patRed)) {
    int initLen = str.length();
    str = regex_replace(str, redRpl, "R");
    count += (initLen - str.length());
  }
  while (regex_match(str, patBlu)) {
    int initLen = str.length();
    str = regex_replace(str, bluRpl, "B");
    count += (initLen - str.length());
  }
  while (regex_match(str, patGre)) {
    int initLen = str.length();
    str = regex_replace(str, greRpl, "G");
    count += (initLen - str.length());
  }

  cout << count << endl;

  return 0;
}
