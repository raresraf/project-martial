#include <iostream>
#include <string>
using namespace std;

int main() {
  int no;
  int output = 0;
  char lastcolor;
  char temp;
  char color;
  string colors;
  cin >> no;
  cin >> colors;
  if (no > 1) {
    color = colors.at(1);
    for (int i = 1; i < no; i++) {
      color = colors.at(i);
      lastcolor = colors.at(i - 1);
      if (lastcolor == color) {
        output++;
      }
    }
  }
  cout << output;
  return 0;
}
