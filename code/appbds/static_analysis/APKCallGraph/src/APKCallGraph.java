```java
import java.io.*;
import java.sql.*;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.jgrapht.ext.ComponentNameProvider;
import org.jgrapht.ext.DOTExporter;
import org.jgrapht.graph.DirectedPseudograph;
import org.xmlpull.v1.XmlPullParserException;

import soot.*;
import soot.jimple.InvokeExpr;
import soot.jimple.Stmt;
import soot.jimple.infoflow.android.SetupApplication;
import soot.options.Options;
import soot.PhaseOptions;
import soot.tagkit.LineNumberTag;
import soot.util.Chain;
import soot.util.queue.QueueReader;

/**
 * APKCallGraph - The class that generates a call graph for an APK, with extended method body information.
 */
public class APKCallGraph {

    static DirectedPseudograph<APKCallGraph.MethodNode, CallEdge> jg =
            new DirectedPseudograph<>(CallEdge.class);

    static HashMap<String, APKCallGraph.MethodNode> methods = new HashMap<>();
    static HashMap<SootMethod, Boolean> visited = new HashMap<SootMethod, Boolean>();
    static ArrayList<SootMethod> methodsList = new ArrayList<>();
    static APKCallGraph apkg = new APKCallGraph();
    static HashMap<String, String> actlayout = new HashMap<>();

    static ArrayList<SootMethod> handleMessageMethods = new ArrayList<>();
    static ArrayList<SootMethod> asyncExecuteMethods = new ArrayList<>();
    static ArrayList<SootMethod> clickMethods = new ArrayList<>();
    static ArrayList<SootMethod> threadTgt = new ArrayList<>();

    static ArrayList<String> threadSrc = new ArrayList<>();

    static int edgeId = 0;
    static int nodeId = 0;
    static boolean isGenerated = false;

    static IC3ProtobufParser ic3parser = new IC3ProtobufParser();

    static HashMap<String, List<String>> edges = new HashMap<>();
    static HashMap<String, List<String>> afterICC = new HashMap<>();

    static HashMap<String, ArrayList<String>> lineVSHdl = new HashMap<>();
    static HashMap<String, ArrayList<String>> HdlVSPM = new HashMap<>();
    static HashMap<String, ArrayList<String>> permMethods = new HashMap<>();
    static ArrayList<String> handlers = new ArrayList<>();
    static HashMap<String, ArrayList<Stmt>> methodToStmts = new HashMap<>();

    // Replace with a suitable path to your local Android jar if needed
    static String androidPlatformPath = "/path/to/android.jar";

    /**
     * MethodNode represents a method node in the call graph, containing the method signature and body information.
     */
    class MethodNode {
        SootMethod m;
        String signature;
        public int id;
        private String jimpleBody;

        public MethodNode(SootMethod m, int id) {
            this.m = m;
            this.id = id;
            if (m != null) {
                signature = m.getSignature();
                loadJimpleBody();
            }
        }

        public MethodNode(String name, int id) {
            this.signature = name;
            this.id = id;
            this.jimpleBody = "// No method information available";
        }

        /**
         * Safely load the method body; if unavailable, set placeholder information.
         */
        private void loadJimpleBody() {
            try {
                if (m.isConcrete() && m.hasActiveBody()) {
                    Body body = m.retrieveActiveBody();
                    jimpleBody = formatJimpleBody(body.toString());
                } else {
                    jimpleBody = generatePlaceholderBody();
                }
            } catch (Exception e) {
                System.err.println("Error loading body for method: " + signature + " - " + e.getMessage());
                jimpleBody = generatePlaceholderBody();
            }
        }

        /**
         * Format Jimple method body, remove any HTML tags, and keep real line breaks to preserve code structure.
         */
        private String formatJimpleBody(String rawBody) {
            if (rawBody == null || rawBody.trim().isEmpty()) {
                return "// Empty method body";
            }
            String[] lines = rawBody.split("\n");
            StringBuilder formatted = new StringBuilder();
            formatted.append("// Method body length: ").append(lines.length).append(" lines\n");
            for (String line : lines) {
                line = line.trim();
                if (!line.isEmpty()) {
                    formatted.append("    ").append(line).append("\n");
                }
            }
            return formatted.toString();
        }

        /**
         * Generate placeholder method body information (without using HTML tags).
         */
        private String generatePlaceholderBody() {
            if (m == null) {
                return "// No method information available";
            }
            StringBuilder info = new StringBuilder();
            info.append("// Method type: ");
            if (m.isAbstract()) {
                info.append("abstract\n");
            } else if (m.isNative()) {
                info.append("native\n");
            } else if (m.isPhantom()) {
                info.append("phantom\n");
            } else {
                info.append("concrete\n");
            }
            info.append("// Declaring class: ").append(m.getDeclaringClass().getName()).append("\n");
            info.append("// Return type: ").append(m.getReturnType()).append("\n");
            info.append("// Parameters: ").append(m.getParameterTypes()).append("\n");
            return info.toString();
        }

        public String getSignature() {
            return signature;
        }

        public String getJimpleBody() {
            return jimpleBody;
        }

        @Override
        public int hashCode() {
            final int prime = 31;
            int result = 1;
            result = prime * result + getOuterType().hashCode();
            result = prime * result + ((signature == null) ? 0 : signature.hashCode());
            return result;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj)
                return true;
            if (obj == null)
                return false;
            if (getClass() != obj.getClass())
                return false;
            MethodNode other = (MethodNode) obj;
            if (!getOuterType().equals(other.getOuterType()))
                return false;
            if (signature == null) {
                if (other.signature != null)
                    return false;
            } else if (!signature.equals(other.signature))
                return false;
            return true;
        }

        private APKCallGraph getOuterType() {
            return APKCallGraph.this;
        }
    }

    /**
     * CallEdge represents an edge in the call graph, indicating a call from one method node to another.
     */
    class CallEdge {
        private MethodNode source;
        private MethodNode target;
        private int id;

        public CallEdge(MethodNode source, MethodNode target, int id) {
            super();
            this.source = source;
            this.target = target;
            this.id = id;
        }

        public MethodNode getSource() {
            return source;
        }

        public void setSource(MethodNode source) {
            this.source = source;
        }

        public MethodNode getTarget() {
            return target;
        }

        public void setTarget(MethodNode target) {
            this.target = target;
        }

        public int getId() {
            return id;
        }

        public void setId(int id) {
            this.id = id;
        }

        @Override
        public String toString() {
            return "Edge " + id + ": " + source.getSignature() + " -> " + target.getSignature();
        }
    }

    /**
     * MethodNodeNameProvider supplies node labels for DOTExporter output.
     * Only method signature and a plain-text method body are retained for further script parsing.
     */
    class MethodNodeNameProvider implements ComponentNameProvider<MethodNode> {

        @Override
        public String getName(MethodNode e) {
            // Combine signature and body in plain text form
            StringBuilder sb = new StringBuilder();
            sb.append("Signature: ").append(e.getSignature()).append("\n");
            sb.append("Body:\n").append(e.getJimpleBody());
            return escapeForDot(sb.toString());
        }

        /**
         * Escape characters that may conflict with DOT labels (quotes, backslashes, newlines).
         * Newlines are replaced with '\l' so that Graphviz can correctly break lines.
         */
        private String escapeForDot(String text) {
            if (text == null) return "";
            text = text.replace("\\", "\\\\");
            text = text.replace("\"", "\\\"");
            text = text.replace("\n", "\\l");
            return text;
        }
    }

    /**
     * CallEdgeLabelProvider supplies edge labels for DOTExporter output.
     */
    class CallEdgeLabelProvider implements ComponentNameProvider<CallEdge> {

        @Override
        public String getName(CallEdge e) {
            return e.toString();
        }
    }

    /**
     * MethodnodeIdProvider supplies node IDs for the DOTExporter.
     */
    class MethodnodeIdProvider implements ComponentNameProvider<MethodNode> {
        @Override
        public String getName(MethodNode e) {
            return "" + e.id;
        }
    }

    /**
     * Main function. Basic configuration, then generate the call graph from the provided APK.
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.out.println("Usage: java APKCallGraph <apk file path> <ic3 output path> <android sdk path>");
            System.exit(1);
        }

        String apkFilePath = args[0];
        String ic3OutputPath = args[1];
        androidPlatformPath = args[2];

        File apkFile = new File(apkFilePath);
        if (!apkFile.exists() || !apkFile.isFile()) {
            System.out.println("Error: APK file does not exist: " + apkFilePath);
            System.exit(1);
        }

        File ic3Dir = new File(ic3OutputPath);
        if (!ic3Dir.exists() || !ic3Dir.isDirectory()) {
            System.out.println("Error: IC3 output directory does not exist: " + ic3OutputPath);
            System.exit(1);
        }

        File sdkDir = new File(androidPlatformPath);
        if (!sdkDir.exists() || !sdkDir.isFile()) {
            System.out.println("Error: Invalid Android SDK path: " + androidPlatformPath);
            System.exit(1);
        }

        String apk = apkFilePath.substring(apkFilePath.lastIndexOf(File.separator) + 1);
        apk = apk.substring(0, apk.indexOf(".apk"));

        System.out.println("Start analyzing APK: " + apk);

        generateCallGraph(apk, apkFilePath, ic3OutputPath);

        File mappingDir = new File("./activity_id_mapping/");
        if (!mappingDir.exists()) {
            mappingDir.mkdirs();
        }

        File mappingfile = new File("./activity_id_mapping/" + apk + ".txt");
        if (!mappingfile.exists()) {
            mappingfile.createNewFile();
        }

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(mappingfile))) {
            for (String key : actlayout.keySet()) {
                bw.write(key + "\t" + actlayout.get(key) + "\n");
            }
        }
    }

    /**
     * Generate the call graph for the given APK.
     */
    public static void generateCallGraph(String apk, String apkPath, String ic3Path)
            throws IOException, XmlPullParserException {

        SetupApplication app = new SetupApplication(androidPlatformPath, apkPath);
        Options.v().set_android_api_version(18);
        app.calculateSourcesSinksEntrypoints("./SourcesAndSinks.txt");
        soot.G.reset();
        Options.v().set_keep_line_number(true);
        Options.v().set_src_prec(Options.src_prec_apk);
        Options.v().set_process_dir(Collections.singletonList(apkPath));
        Options.v().set_force_android_jar(androidPlatformPath);
        Options.v().setPhaseOption("cg.spark", "on");
        Options.v().setPhaseOption("cg.cha", "enabled:true");
        Options.v().setPhaseOption("cg", "all-reachable:true");
        PhaseOptions.v().setPhaseOption("tag.ln", "on");
        Options.v().set_android_api_version(18);
        Options.v().set_whole_program(true);
        Options.v().set_allow_phantom_refs(true);
        Options.v().set_ignore_resolution_errors(true);
        Options.v().set_output_format(Options.output_format_jimple);
        Options.v().set_no_bodies_for_excluded(true);

        app.setCallbackFile("./AndroidCallbacks.txt");

        String sootCp = apkPath + File.pathSeparator + androidPlatformPath;
        Options.v().set_soot_classpath(sootCp);
        Options.v().set_process_multiple_dex(true);
        Options.v().set_include_all(true);

        Scene.v().loadNecessaryClasses();

        for (SootClass sc : Scene.v().getApplicationClasses()) {
            sc.setApplicationClass();
        }

        SootMethod entryPoint = app.getEntryPointCreator().createDummyMain();
        Options.v().set_main_class(entryPoint.getSignature());
        Scene.v().setEntryPoints(Collections.singletonList(entryPoint));

        PackManager.v().runPacks();
        Chain<SootClass> applicationClasses = Scene.v().getApplicationClasses();
        for (SootClass sootClass : applicationClasses) {
            List<SootMethod> ms = sootClass.getMethods();
            for (SootMethod m : ms) {
                if (methodsList.contains(m)) {
                    continue;
                }
                methodsList.add(m);
            }
        }

        CopyOnWriteArrayList<SootMethod> list = new CopyOnWriteArrayList<>(methodsList);
        APKCallGraph instance = new APKCallGraph();

        while (list.size() > 0) {
            SootMethod currentMethod = list.get(0);
            list.remove(0);

            // Handle Activity's onCreate
            if (currentMethod.toString().contains("onCreate(android.os.Bundle)")) {
                if (currentMethod.getDeclaringClass().getSuperclass().toString().contains("Activity")) {
                    try {
                        if (!currentMethod.hasActiveBody()) {
                            System.err.println("Method is not concrete: " + currentMethod.getSignature());
                            continue;
                        }
                        Body body = currentMethod.retrieveActiveBody();
                        Iterator<Unit> stmts = body.getUnits().iterator();
                        visited.put(currentMethod, true);
                        while (stmts.hasNext()) {
                            Stmt s = (Stmt) stmts.next();
                            if (s.containsInvokeExpr()) {
                                InvokeExpr expr = s.getInvokeExpr();
                                if (expr.getMethod().getName().equals("setContentView")) {
                                    if (expr.getArgCount() > 0) {
                                        Value arg = expr.getArg(0);
                                        String id = arg.toString();
                                        actlayout.put(currentMethod.getDeclaringClass().toString(), id);
                                        System.out.println("Activity: " +
                                                currentMethod.getDeclaringClass().toString());
                                        System.out.println("Layout ID: " + id);
                                    } else {
                                        System.err.println("setContentView method has no arguments in: " +
                                                currentMethod.getSignature());
                                    }
                                }
                            }
                        }
                    } catch (Exception e) {
                        System.err.println("Error processing onCreate method: " + currentMethod.getSignature());
                        e.printStackTrace();
                        continue;
                    }
                }
            }

            // Handle Fragment's onCreateView
            if (currentMethod.toString().contains("onCreateView(")) {
                if (currentMethod.getDeclaringClass().getSuperclass().toString().contains("Fragment")) {
                    try {
                        if (!currentMethod.hasActiveBody()) {
                            System.err.println("Method is not concrete: " + currentMethod.getSignature());
                            continue;
                        }
                        Body body = currentMethod.retrieveActiveBody();
                        Iterator<Unit> stmts = body.getUnits().iterator();
                        visited.put(currentMethod, true);
                        while (stmts.hasNext()) {
                            Stmt s = (Stmt) stmts.next();
                            if (s.containsInvokeExpr()) {
                                InvokeExpr expr = s.getInvokeExpr();
                                if (expr.getMethod().getName().equals("inflate")) {
                                    if (expr.getArgCount() > 0) {
                                        Value arg = expr.getArg(0);
                                        String id = arg.toString();
                                        actlayout.put(currentMethod.getDeclaringClass().toString(), id);
                                        System.out.println("Fragment: " +
                                                currentMethod.getDeclaringClass().toString());
                                        System.out.println("Layout ID: " + id);
                                    } else {
                                        System.err.println("inflate method has no arguments in: " +
                                                currentMethod.getSignature());
                                    }
                                }
                            }
                        }
                    } catch (Exception e) {
                        System.err.println("Error processing onCreateView method: " + currentMethod.getSignature());
                        e.printStackTrace();
                        continue;
                    }
                }
            }

            // Handle other methods
            if (!visited.containsKey(currentMethod)) {
                if (!currentMethod.hasActiveBody()) {
                    continue;
                }
                try {
                    Body body = currentMethod.retrieveActiveBody();
                    Iterator<Unit> stmts = body.getUnits().iterator();
                    visited.put(currentMethod, true);
                    while (stmts.hasNext()) {
                        Stmt s = (Stmt) stmts.next();
                        if (s.containsInvokeExpr()) {
                            String signature = currentMethod.getSignature();

                            if (signature.contains("void handleMessage(android.os.Message)>")) {
                                if (!handleMessageMethods.contains(currentMethod)) {
                                    handleMessageMethods.add(currentMethod);
                                }
                            }
                            if (signature.contains("doInBackground(")
                                    || signature.contains("onPreExecute(")
                                    || signature.contains("onPostExecute(")) {
                                if (!asyncExecuteMethods.contains(currentMethod)) {
                                    asyncExecuteMethods.add(currentMethod);
                                }
                            }
                            if (signature.contains(": void onClick(")) {
                                if (!clickMethods.contains(currentMethod)) {
                                    clickMethods.add(currentMethod);
                                }
                            }
                            if (signature.contains(": void run()")) {
                                if (currentMethod.getDeclaringClass()
                                        .getSuperclass().toString().contains("java.lang.Thread")) {
                                    if (!threadTgt.contains(currentMethod)) {
                                        threadTgt.add(currentMethod);
                                        String threadSrcMethod = "<"
                                                + currentMethod.getDeclaringClass().toString()
                                                + ": void start()>";
                                        if (!threadSrc.contains(threadSrcMethod)) {
                                            threadSrc.add(threadSrcMethod);
                                        }
                                    }
                                }
                            }

                            try {
                                InvokeExpr expr = s.getInvokeExpr();
                                String invokedMethodSignature = expr.getMethod().getSignature();
                                if (expr.getMethod().getDeclaringClass().toString()
                                        .equals("java.lang.Thread")) {
                                    invokedMethodSignature = expr.getMethodRef().getSignature();
                                }
                                edges.computeIfAbsent(signature, k -> new ArrayList<>());
                                if (!edges.get(signature).contains(invokedMethodSignature)) {
                                    edges.get(signature).add(invokedMethodSignature);
                                }
                                if (!methodsList.contains(expr.getMethod())) {
                                    methodsList.add(expr.getMethod());
                                    list.add(expr.getMethod());
                                }

                                methodToStmts
                                        .computeIfAbsent(expr.getMethod().getSignature(), k -> new ArrayList<>())
                                        .add(s);
                            } catch (Exception e) {
                                System.err.println("Error retrieving InvokeExpr for method: "
                                        + currentMethod.getSignature());
                                e.printStackTrace();
                            }
                        }
                    }
                } catch (Exception e) {
                    System.err.println("Error processing method: " + currentMethod.getSignature());
                    e.printStackTrace();
                    continue;
                }
            }
        }

        // Construct the call graph (supports placeholder nodes)
        for (String src : edges.keySet()) {
            for (String tgt : edges.get(src)) {
                edgeId++;

                // Create or retrieve source node
                MethodNode srcNode;
                if (methods.containsKey(src)) {
                    srcNode = methods.get(src);
                } else {
                    SootMethod srcMethod = findMethodBySignature(src);
                    if (srcMethod == null) {
                        nodeId++;
                        srcNode = apkg.new MethodNode(src, nodeId);
                        methods.put(src, srcNode);
                        jg.addVertex(srcNode);
                        System.out.println("Created placeholder node for: " + src);
                    } else {
                        nodeId++;
                        srcNode = apkg.new MethodNode(srcMethod, nodeId);
                        methods.put(src, srcNode);
                        jg.addVertex(srcNode);
                    }
                }

                // Create or retrieve target node
                MethodNode tgtNode;
                if (methods.containsKey(tgt)) {
                    tgtNode = methods.get(tgt);
                } else {
                    SootMethod tgtMethod = findMethodBySignature(tgt);
                    if (tgtMethod == null) {
                        nodeId++;
                        tgtNode = apkg.new MethodNode(tgt, nodeId);
                        methods.put(tgt, tgtNode);
                        jg.addVertex(tgtNode);
                        System.out.println("Created placeholder node for: " + tgt);
                    } else {
                        nodeId++;
                        tgtNode = apkg.new MethodNode(tgtMethod, nodeId);
                        methods.put(tgt, tgtNode);
                        jg.addVertex(tgtNode);
                    }
                }

                // Add edge
                jg.addEdge(srcNode, tgtNode, apkg.new CallEdge(srcNode, tgtNode, edgeId));
            }
        }

        boolean useic3 = true;
        if (useic3) {
            File ic3folder = new File(ic3Path + File.separator + apk);
            File[] listFiles = ic3folder.listFiles();
            if (listFiles == null) {
                System.out.println("No IC3 output found at: " + ic3Path + File.separator + apk);
            }
            if (listFiles != null && listFiles.length > 0) {
                for (File listfile : listFiles) {
                    HashMap<String, HashSet<String>> m2providers = new HashMap<>();
                    HashMap<String, HashSet<String>> m2intents = new HashMap<>();
                    HashMap<String, ArrayList<String>> iccs =
                            ic3parser.parseFromFile(listfile.getAbsolutePath(), m2providers, m2intents);

                    for (Entry<String, ArrayList<String>> icc : iccs.entrySet()) {
                        String fromMethod = icc.getKey();
                        ArrayList<String> toClasses = icc.getValue();

                        if (!methods.containsKey(fromMethod)) {
                            System.out.println("Error: fromMethod not found: " + fromMethod);
                            continue;
                        }

                        MethodNode from = methods.get(fromMethod);
                        for (String clazz : toClasses) {
                            if (clazz.length() == 0) {
                                continue;
                            }
                            try {
                                SootClass loadClass = Scene.v().loadClassAndSupport(clazz);
                                loadClass.setApplicationClass();
                                List<SootMethod> loadMethods = loadClass.getMethods();
                                for (SootMethod loadMethod : loadMethods) {
                                    if (loadMethod.getName().startsWith("onCreate")
                                            || loadMethod.getName().startsWith("onStart")) {
                                        if (!methods.containsKey(loadMethod.getSignature())) {
                                            nodeId++;
                                            MethodNode loadNode = apkg.new MethodNode(loadMethod, nodeId);
                                            methods.put(loadMethod.getSignature(), loadNode);
                                            jg.addVertex(loadNode);
                                        }
                                        // If needed, edges can be added here
                                        // jg.addEdge(from, methods.get(loadMethod.getSignature()),
                                        //     apkg.new CallEdge(from, methods.get(loadMethod.getSignature()), edgeId++));
                                    }
                                }
                            } catch (Exception e) {
                                System.err.println("Error loading class for ICC: " + clazz + " - " + e.getMessage());
                                continue;
                            }
                        }

                        for (String src : edges.getOrDefault(fromMethod, Collections.emptyList())) {
                            if (!src.contains(": void startActivity(")) {
                                continue;
                            } else {
                                if (icc.getValue().isEmpty()) {
                                    System.out.println("No target classes for ICC from method: " + fromMethod);
                                    continue;
                                }
                                String tgt = "<" + icc.getValue().get(0) + ": void onCreate(android.os.Bundle)>";
                                if (jg.containsVertex(methods.get(tgt))) {
                                    afterICC.computeIfAbsent(fromMethod, k -> new ArrayList<>()).add(tgt);
                                    edges.computeIfAbsent(src, k -> new ArrayList<>()).add(tgt);
                                    jg.addEdge(methods.get(src), methods.get(tgt),
                                            apkg.new CallEdge(methods.get(src), methods.get(tgt), edgeId++));
                                } else {
                                    System.out.println("Target method not found in graph: " + tgt);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Connect threads, handleMessage, async, etc.
        for (String m : methods.keySet()) {
            connectThread(m);
            connectSendMessage(m);
            connectAsyncExecute(m);
            connectClickCall(m);
        }

        // Export call graph in DOT format
        DOTExporter<MethodNode, CallEdge> exporter = new DOTExporter<>(
                apkg.new MethodnodeIdProvider(),
                apkg.new MethodNodeNameProvider(),
                apkg.new CallEdgeLabelProvider()
        );
        File outputDir = new File("./dot_output/" + apk + "/");
        if (!outputDir.exists()) {
            boolean dirCreated = outputDir.mkdirs();
            if (!dirCreated) {
                System.err.println("Failed to create output directory: " + outputDir.getAbsolutePath());
                return;
            }
        }
        try (FileWriter writer = new FileWriter(new File(outputDir, apk + ".dot"))) {
            exporter.exportGraph(jg, writer);
            System.out.println("Call graph exported to: "
                    + new File(outputDir, apk + ".dot").getAbsolutePath());
        } catch (IOException e) {
            System.err.println("Error exporting call graph: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Find a SootMethod by its signature.
     *
     * @param signature The method signature
     * @return The corresponding SootMethod object, or null if not found
     */
    public static SootMethod findMethodBySignature(String signature) {
        for (SootClass sootClass : Scene.v().getApplicationClasses()) {
            for (SootMethod method : sootClass.getMethods()) {
                if (method.getSignature().equals(signature)) {
                    return method;
                }
            }
        }
        // If not found in application classes, try library classes
        try {
            String className = signature.substring(1, signature.indexOf(":"));
            SootClass sootClass = Scene.v().loadClassAndSupport(className);

            for (SootMethod method : sootClass.getMethods()) {
                if (method.getSignature().equals(signature)) {
                    return method;
                }
            }
        } catch (Exception e) {
            System.err.println("Error loading class for method: " + signature);
        }
        return null;
    }

    public static void connectThread(String src) {
        if (threadSrc.contains(src)) {
            MethodNode srcNode = methods.get(src);
            if (srcNode == null) {
                System.err.println("Source node not found for: " + src);
                return;
            }
            for (SootMethod tgt : threadTgt) {
                if (tgt.getSignature().contains(src.substring(0, src.indexOf(":")))) {
                    MethodNode tgtNode = methods.get(tgt.getSignature());
                    if (tgtNode == null) {
                        System.err.println("Target node not found for method: " + tgt.getSignature());
                        continue;
                    }
                    edges.computeIfAbsent(src, k -> new ArrayList<>()).add(tgt.getSignature());
                    jg.addEdge(srcNode, tgtNode, apkg.new CallEdge(srcNode, tgtNode, edgeId++));
                }
            }
            System.out.println("Connect thread successful");
        }
    }

    public static void connectClickCall(String src) {
        if (src.contains(": void setOnClickListener(")) {
            for (SootMethod tgt : clickMethods) {
                MethodNode srcNode = methods.get(src);
                MethodNode tgtNode = methods.get(tgt.getSignature());
                if (srcNode == null || tgtNode == null) {
                    System.err.println("Source or Target node not found for: " + src
                            + " -> " + tgt.getSignature());
                    continue;
                }
                edges.computeIfAbsent(src, k -> new ArrayList<>()).add(tgt.getSignature());
                jg.addEdge(srcNode, tgtNode, apkg.new CallEdge(srcNode, tgtNode, edgeId++));
            }
            System.out.println("Connect click calls successful");
        }
    }

    public static void connectSendMessage(String src) {
        if (src.contains("boolean sendMessage(android.os.Message)>")) {
            for (SootMethod tgt : handleMessageMethods) {
                MethodNode srcNode = methods.get(src);
                MethodNode tgtNode = methods.get(tgt.getSignature());
                if (srcNode == null || tgtNode == null) {
                    System.err.println("Source or Target node not found for: " + src
                            + " -> " + tgt.getSignature());
                    continue;
                }
                edges.computeIfAbsent(src, k -> new ArrayList<>()).add(tgt.getSignature());
                jg.addEdge(srcNode, tgtNode, apkg.new CallEdge(srcNode, tgtNode, edgeId++));
            }
            System.out.println("Connect message calls successful");
        }
    }

    public static void connectAsyncExecute(String src) {
        if (src.contains("android.os.AsyncTask execute(java.lang.Object[])>")
                || src.contains("AsyncTask executeOnExecutor(")) {
            for (SootMethod tgt : asyncExecuteMethods) {
                MethodNode srcNode = methods.get(src);
                MethodNode tgtNode = methods.get(tgt.getSignature());
                if (srcNode == null || tgtNode == null) {
                    System.err.println("Source or Target node not found for: " + src
                            + " -> " + tgt.getSignature());
                    continue;
                }
                edges.computeIfAbsent(src, k -> new ArrayList<>()).add(tgt.getSignature());
                jg.addEdge(srcNode, tgtNode, apkg.new CallEdge(srcNode, tgtNode, edgeId++));
            }
            System.out.println("Connect asynctask calls successful");
        }
    }

    /**
     * Generate a subgraph of the specified method and export it as a DOT file.
     */
    public static void generateSubGraphOfMethod(String apk, String method)
            throws IOException, SQLException {
        DirectedPseudograph<APKCallGraph.MethodNode, CallEdge> subGraph =
                new DirectedPseudograph<>(CallEdge.class);

        ArrayList<String> subgraph = new ArrayList<>();
        if (!methods.containsKey(method)) {
            System.out.println("Method not found, please enter again!");
        } else {
            HashMap<String, Boolean> hasVisited = new HashMap<>();
            List<String> list = new ArrayList<>();
            list.add(method);
            hasVisited.put(method, true);
            while (list.size() > 0) {
                String current = list.get(0);
                if (edges.get(current) == null) {
                    list.remove(0);
                    continue;
                }

                for (String tgt : edges.get(current)) {
                    if (current.contains(": void startActivity(")) {
                        if (afterICC.get(method) == null) {
                            continue;
                        }
                    }
                    if (!subGraph.containsVertex(methods.get(current))) {
                        subGraph.addVertex(methods.get(current));
                    }
                    if (!subGraph.containsVertex(methods.get(tgt))) {
                        subGraph.addVertex(methods.get(tgt));
                    }
                    subGraph.addEdge(
                            methods.get(current),
                            methods.get(tgt),
                            apkg.new CallEdge(methods.get(current), methods.get(tgt), edgeId++)
                    );

                    if (hasVisited.containsKey(tgt)) {
                        continue;
                    }
                    hasVisited.put(tgt, true);
                    list.add(tgt);
                    subgraph.add(tgt);
                }
                list.remove(0);
            }

            DOTExporter<MethodNode, CallEdge> exporter = new DOTExporter<>(
                    apkg.new MethodnodeIdProvider(),
                    apkg.new MethodNodeNameProvider(),
                    null
            );
            File tempDir = new File("./temp/" + apk + "/");
            if (!tempDir.exists() && !tempDir.mkdirs()) {
                System.err.println("Failed to create temp directory: " + tempDir.getAbsolutePath());
                return;
            }
            exporter.exportGraph(
                    subGraph,
                    new FileWriter("./temp/" + apk + "/" + method + ".dot")
            );

            isGenerated = true;
        }
    }

    /**
     * Retrieve permission information for a method (example uses MySQL).
     * Modify with appropriate connection details.
     */
    public static String getPermission(String method) throws SQLException {
        Connection connection = null;
        String permission = null;
        String methodClass = method.substring(1, method.indexOf(":"));
        String sql = "select Permission from outputmapping where Method = '" + method + "'";
        String driver = "com.mysql.cj.jdbc.Driver";
        String url = "your mysql url";
        try {
            Class.forName(driver);
            connection = DriverManager.getConnection(url);
            Statement stmt = connection.createStatement();
            ResultSet resultSet = stmt.executeQuery(sql);
            if (resultSet.next()) {
                permission = resultSet.getString(1);
            }
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        } finally {
            if (connection != null && !connection.isClosed()) {
                connection.close();
            }
        }
        return permission;
    }

    /**
     * Parse handlers from a file and generate subgraphs.
     */
    public static void getHandlers(String apk, String fileName) throws IOException, SQLException {
        System.out.println("Processing file :" + fileName);
        try {
            String regex = "<(.*?)>";
            FileReader f_reader = new FileReader(fileName);
            BufferedReader br = new BufferedReader(f_reader);
            String line = "";
            while ((line = br.readLine()) != null) {
                String tempLine = "";
                Pattern pattern = Pattern.compile(regex);
                Matcher matcher = pattern.matcher(line);
                while (matcher.find()) {
                    tempLine = line.substring(0, line.indexOf("["));
                    if (handlers.contains(matcher.group(1))) {
                        if (!lineVSHdl.containsKey(tempLine)) {
                            ArrayList<String> temp = new ArrayList<>();
                            temp.add("<" + matcher.group(1) + ">");
                            lineVSHdl.put(tempLine, temp);
                            continue;
                        } else {
                            lineVSHdl.get(tempLine).add("<" + matcher.group(1) + ">");
                            continue;
                        }
                    }
                    handlers.add(matcher.group(1));
                    if (!lineVSHdl.containsKey(tempLine)) {
                        ArrayList<String> temp = new ArrayList<>();
                        temp.add("<" + matcher.group(1) + ">");
                        lineVSHdl.put(tempLine, temp);
                        generateSubGraphOfMethod(apk, "<" + matcher.group(1) + ">");
                    } else {
                        lineVSHdl.get(tempLine).add("<" + matcher.group(1) + ">");
                        generateSubGraphOfMethod(apk, "<" + matcher.group(1) + ">");
                    }
                }
            }
            br.close();
        } catch (Exception e) {
            System.out.println(e.toString());
        }
    }

    /**
     * Write information to a CSV file.
     */
    public static void writeInfoToFile(String permissionOutput, String apkName) throws IOException {
        File file = new File(permissionOutput + apkName + "_permission.csv");
        if (!file.exists()) {
            file.createNewFile();
        }
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file))) {
            bw.write("APK\tImage\tWID\tWID Name\tLayout\tHandler\tMethod\tLines\tPermissions\n");
            for (String line : lineVSHdl.keySet()) {
                for (String handler : HdlVSPM.keySet()) {
                    if (!lineVSHdl.get(line).contains(handler)) {
                        continue;
                    }
                    for (String method : permMethods.keySet()) {
                        for (SootMethod m : methodsList) {
                            if (m.toString().equals(method)) {
                                ArrayList<Stmt> stmts = methodToStmts.get(m.getSignature());
                                ArrayList<String> lineNums = new ArrayList<>();
                                if (stmts != null) {
                                    for (Stmt s : stmts) {
                                        Unit u = (Unit) s;
                                        LineNumberTag tag = (LineNumberTag) u.getTag("LineNumberTag");
                                        if (tag != null) {
                                            lineNums.add(String.valueOf(tag.getLineNumber()));
                                        }
                                    }
                                }
                                bw.write(line + handler + "\t"
                                        + method + "\t"
                                        + lineNums + "\t"
                                        + permMethods.get(method) + "\n");
                            }
                        }
                    }
                }
            }
        }
    }
}
```