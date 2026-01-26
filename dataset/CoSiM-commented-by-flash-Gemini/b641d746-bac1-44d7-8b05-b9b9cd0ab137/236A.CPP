#include <cstring>
#include <iostream>
#include <set>
using namespace std;
int main() {
  set<char> a;
  char b[101];
  int i;
  fgets(b, sizeof(b), stdin);
  for (i = 0; i < strlen(b); i++) {
    a.insert(b[i]);
  }
  i = a.size();
  if (i % 2 == 0)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";
  return 0;
}