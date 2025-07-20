#include <iostream>
#include <set>
using namespace std;

int main(int argc, char *argv[]) {
  set<char> letters;
  string s;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    letters.insert(s.at(i));
  }
  if (letters.size() % 2 == 0)
    printf("CHAT WITH HER!\n");
  else
    printf("IGNORE HIM!\n");
  return 0;
}
