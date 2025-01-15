#include <cstring>
#include <iostream>
using namespace std;
int main() {
  char str[1000];
  cin.getline(str, 1000);
  int i;
  int n = strlen(str);
  str[0] = toupper(str[0]);
  for (i = 0; str[i] != '\0'; i++) {
    if (str[i] == ' ') {

      str[i + 1] = toupper(str[i + 1]);
    }
  }
  cout.write(str, n);
  return 0;
}
