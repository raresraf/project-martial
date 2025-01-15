#include <iostream>
using namespace std;

int main(int argc, char *argv[]) {
  int luckyCount = 0;
  string input;
  cin >> input;
  for (int i = 0; i < input.length(); i++) {
    if (input.at(i) == '4' || input.at(i) == '7')
      luckyCount++;
  }
  if (luckyCount == 4 || luckyCount == 7)
    printf("YES\n");
  else
    printf("NO\n");
  return 0;
}
