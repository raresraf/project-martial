#include <algorithm>
#include <iostream>
#include <set>
#include <vector>

using namespace std;

int main(int argc, char const *argv[]) {
  set<char> s;
  char c;
  while (true) {
    scanf("%c", &c);
    if (c == '\n')
      break;

    s.insert(c);
  }
  if (s.size() % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
  return 0;
}
