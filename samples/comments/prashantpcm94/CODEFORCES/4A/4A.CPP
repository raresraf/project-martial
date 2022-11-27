import java.util.Scanner;
public
class watermelon {
public
  static void main(String args[]) {
    int n;
    Scanner sc = new Scanner(System.in);
    n = sc.nextInt();
    if (n == 2 || n % 2 != 0)
      System.out.println("NO");
    else
      System.out.println("YES");
  }
}