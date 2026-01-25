#include <cstdio>
#include <iostream>
#include <list>
using namespace std;

int main(int argc, char *argv[]) {
  string s;
  list<char> colors;
  char prev;
  int count, ans = 0;

  cin >> count >> s;

  if (count == 1) {
    printf("0\n");
    return 0;
  }

  for (int i = 0; i < count; i++) {
    colors.push_back(s.at(i));
  }

  list<char>::iterator it;
  it = colors.begin();
  prev = *it;
  colors.pop_front();
  for (it = colors.begin(); it != colors.end(); ++it) {
    if (*it == prev) {
      ans++;
    }
    prev = *it;
  }

  printf("%d\n", ans);
  return 0;
}
