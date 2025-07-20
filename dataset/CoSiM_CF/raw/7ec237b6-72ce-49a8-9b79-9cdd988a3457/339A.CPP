#include <iostream>

void parse(std::string &s, int counts[]) {
  for (int i = 0; i < s.size(); ++i) {
    if (s[i] == '1')
      ++counts[0];
    else if (s[i] == '2')
      ++counts[1];
    else if (s[i] == '3')
      ++counts[2];
  }
  // std::cout << counts[0] << '\n';
}

void printOutput(int counts[]) {
  std::string one = "", two = "", three = "";
  for (int i = 0; i < counts[0]; ++i) {
    one += '1';
    if (i != counts[0] - 1)
      one += '+';
  }
  for (int i = 0; i < counts[1]; ++i) {
    two += '2';
    if (i != counts[1] - 1)
      two += '+';
  }
  for (int i = 0; i < counts[2]; ++i) {
    three += '3';
    if (i != counts[2] - 1)
      three += '+';
  }
  std::string c1 = "", c2 = "";
  if ((one != "") && ((two != "") || (three != "")))
    c1 = "+";
  if ((two != "") && (three != ""))
    c2 = "+";
  std::cout << one << c1 << two << c2 << three << '\n';
}

int main() {
  std::string s;
  std::cin >> s;
  int counts[3] = {0};
  parse(s, counts);
  printOutput(counts);
}
