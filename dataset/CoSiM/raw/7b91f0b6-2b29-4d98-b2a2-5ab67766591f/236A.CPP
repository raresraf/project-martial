/*
        a.c executes with same runtime, but takes much lesser memory.
        This takes additional 100kb.
*/

#include <cstdio>
#include <unordered_set>

using namespace std;

int main() {
  char ch;
  unordered_set<char> S;
  while ((ch = getchar()) != '\n') {
    S.insert(ch);
  }

  if (S.size() % 2 == 0)
    printf("CHAT WITH HER!\n");
  else
    printf("IGNORE HIM!\n");

  return 0;
}